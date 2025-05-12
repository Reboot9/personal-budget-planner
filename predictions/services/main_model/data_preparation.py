import json
import logging
import pathlib

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from predictions.services.main_model.utils import get_logger

from personal_budget_planner.constants import *


class DataPreprocessor:
    """
    Class for processing spending data, including data preparation,
    feature engineering, and category analysis.
    """

    def __init__(self, logger: logging.Logger | None = None):
        self.logger = logger if logger else get_logger(self.__class__.__name__,
                                                       log_file_name='data_formatting.log')
        self.df: pd.DataFrame = pd.DataFrame()

    def _load_and_prepare_initial_df(self, init_data_file_path: pathlib.Path) -> tuple[pd.DataFrame, list[str]]:

        self.logger.info(f"Завантаження початкового набору даних з: {init_data_file_path}")
        try:
            initial_df = pd.read_csv(init_data_file_path)
        except FileNotFoundError:
            self.logger.error(f"Файл не знайдено: {init_data_file_path}")
            raise
        except Exception as e:
            self.logger.error(f"Помилка при читанні CSV файлу {init_data_file_path}: {e}")
            raise

        self.logger.info("Створення часових ознак з дати...")
        initial_df['date'] = pd.to_datetime(initial_df['date'])
        initial_df['month'] = initial_df['date'].dt.month
        initial_df['day_of_week'] = initial_df['date'].dt.dayofweek  # Понеділок=0, Неділя=6
        initial_df['week_of_month'] = (initial_df['date'].dt.day - 1) // 7 + 1
        initial_df['year'] = initial_df['date'].dt.year
        initial_df['day'] = initial_df['date'].dt.day

        initial_df['month_period'] = initial_df[['month', 'year']].apply(lambda x: f'{x.iloc[0]}-{x.iloc[1]}', axis=1)

        self.logger.info(
            f"Роки: {initial_df['year'].nunique()} ({','.join(map(str, sorted(initial_df['year'].unique())))})")
        self.logger.info(f"Послідовні місяці: {initial_df['month_period'].nunique()}")

        categories = list(initial_df["category"].unique())
        self.logger.info(f"Знайдено категорій: {len(categories)}")
        self.logger.info(f"Початкова довжина датафрейму: {len(initial_df)}")
        return initial_df, categories

    def _create_pivot_table(self, initial_df: pd.DataFrame) -> pd.DataFrame:
        """
        Створює зведену таблицю із сумами витрат за категоріями.

        Args:
            initial_df (pd.DataFrame): Початковий DataFrame з необхідними колонками.

        Returns:
            pd.DataFrame: Зведена таблиця.
        """
        self.logger.info("Створення зведеної таблиці...")
        # Зберігаємо оригінальну дату перед перетворенням на dt.date для агрегації
        initial_df_copy = initial_df.copy()  # Робота з копією для уникнення SettingWithCopyWarning
        initial_df_copy['date_agg'] = initial_df_copy['date'].dt.date

        # Використовуємо всі часові ознаки для індексу, щоб зберегти їх
        df_pivoted = initial_df_copy.pivot_table(
            index=['date_agg', 'year', 'month', 'week_of_month', 'day_of_week'],
            columns='category',
            values='amount',
            aggfunc='sum'
        ).reset_index()

        # Перейменовуємо 'date_agg' назад на 'date'
        df_pivoted.rename(columns={'date_agg': 'date'}, inplace=True)

        # Додаємо 'month_period' знову, оскільки він міг бути втрачений при pivot_table, якщо не в індексі
        df_pivoted['month_period'] = df_pivoted[['month', 'year']].apply(lambda x: f'{x.iloc[0]}-{x.iloc[1]}', axis=1)

        df_pivoted.fillna(0, inplace=True)
        self.logger.info(f"Кількість унікальних дат у зведеній таблиці: {len(df_pivoted['date'].unique())}")
        self.logger.info(f"Довжина зведеної таблиці: {len(df_pivoted)}")
        return df_pivoted

    def _classify_spending_categories(self, df_pivoted: pd.DataFrame, categories: list[str],
                                      initial_month_period_nunique: int) -> tuple[
        dict[str, list[str]], list[str], list[str], list[str]]:
        """
        Класифікує категорії витрат на регулярні, періодичні та рідкісні
        на основі кількості нульових значень.

        Args:
            df_pivoted (pd.DataFrame): Зведена таблиця.
            categories (list[str]): Список усіх категорій.
            initial_month_period_nunique (int): Кількість унікальних місячних періодів у початкових даних.

        Returns:
            tuple[dict[str, list[str]], list[str], list[str], list[str]]:
                - Словник з класифікованими колонками.
                - Список регулярних колонок.
                - Список періодичних колонок (above_80_nan).
                - Список рідкісних колонок (above_95_nan).
        """
        self.logger.info("Класифікація категорій витрат за частотою...")
        threshold_80 = 0.80 * len(df_pivoted)
        # Кількість періодів, в яких може не бути витрат взагалі (наприклад, перший і останній місяці неповні)
        # Додаємо SKIPPED_PERIODS, щоб врахувати потенційно пропущені періоди на початку/кінці даних
        threshold_95 = len(df_pivoted) - initial_month_period_nunique + SKIPPED_PERIODS

        # Переконуємося, що всі категорії з initial_df є колонками в df_pivoted
        zero_values = {}
        for category in categories:
            if category in df_pivoted.columns:
                zero_values[category] = len(df_pivoted.loc[df_pivoted[category] == 0])
            else:
                # Якщо категорія взагалі не потрапила в стовпці (малоймовірно після pivot з fillna(0))
                # то всі значення для неї - нулі.
                zero_values[category] = len(df_pivoted)

        zero_values_series = pd.Series(zero_values).sort_values(ascending=True)

        above_95_nan_cols = zero_values_series.loc[zero_values_series > threshold_95].index.to_list()
        above_80_nan_cols = zero_values_series.loc[
            zero_values_series.between(threshold_80, threshold_95, inclusive='right')].index.to_list()
        regular_cols = zero_values_series.loc[zero_values_series < threshold_80].index.to_list()

        self.logger.info(f"Регулярні категорії (<80% нулів): {len(regular_cols)}")
        self.logger.info(f"Періодичні категорії (80-95% нулів): {len(above_80_nan_cols)}")
        self.logger.info(f"Рідкісні категорії (>95% нулів): {len(above_95_nan_cols)}")

        classified_cols = {
            'regular': regular_cols,
            'above_80_nan': above_80_nan_cols,
            'above_95_nan': above_95_nan_cols
        }
        return classified_cols, regular_cols, above_80_nan_cols, above_95_nan_cols

    def _engineer_temporal_features(self, df_target: pd.DataFrame) -> list[str]:
        """
        Створює циклічні часові ознаки (sin, cos) та кодує рік.
        Модифікує DataFrame `df_target` на місці.

        Args:
            df_target (pd.DataFrame): DataFrame для модифікації (очікується self.df).

        Returns:
            list[str]: Список назв нових створених часових колонок.
        """
        self.logger.info("Створення циклічних часових ознак...")
        new_temporal_cols: list[str] = []
        for col, period in TEMPORAL_COLUMNS_CONFIG.items():
            if col not in df_target.columns:
                self.logger.warning(
                    f"Колонка '{col}' не знайдена в DataFrame. Пропускаємо створення циклічних ознак для неї.")
                continue

            values = df_target[col]
            sin_col_name = f'{col}_sin'
            cos_col_name = f'{col}_cos'
            df_target[sin_col_name] = np.sin(2 * np.pi * values / period)
            df_target[cos_col_name] = np.cos(2 * np.pi * values / period)
            new_temporal_cols.extend([sin_col_name, cos_col_name])

        if 'year' in df_target.columns:
            self.logger.info("Кодування року...")
            df_target['encoded_year'] = LabelEncoder().fit_transform(df_target['year'])
            new_temporal_cols.append('encoded_year')
        else:
            self.logger.warning("Колонка 'year' не знайдена для кодування.")

        self.logger.info(f"Створені нові часові колонки: {new_temporal_cols}")
        return new_temporal_cols

    def _engineer_aggregated_spend_features(self, df_target: pd.DataFrame, periodical_cols: list[str],
                                            occasional_cols: list[str]) -> list[str]:
        """
        Створює агреговані ознаки для періодичних та рідкісних витрат.
        Модифікує DataFrame `df_target` на місці.

        Args:
            df_target (pd.DataFrame): DataFrame для модифікації (очікується self.df).
            periodical_cols (list[str]): Список колонок періодичних витрат.
            occasional_cols (list[str]): Список колонок рідкісних витрат.

        Returns:
            list[str]: Список назв нових агрегованих колонок.
        """
        self.logger.info("Створення агрегованих ознак витрат...")
        subtotals_columns: list[str] = []

        if periodical_cols:
            df_target['periodical_sum'] = df_target[periodical_cols].sum(axis=1)
            subtotals_columns.append('periodical_sum')
            self.logger.info("Створено 'periodical_sum'.")
        else:
            df_target['periodical_sum'] = 0  # Якщо немає таких колонок, сума 0
            subtotals_columns.append('periodical_sum')
            self.logger.info("Колонки для 'periodical_sum' відсутні, створено колонку з нулями.")

        if occasional_cols:
            df_target['occasional_sum'] = df_target[occasional_cols].sum(axis=1)
            subtotals_columns.append('occasional_sum')
            self.logger.info("Створено 'occasional_sum'.")
        else:
            df_target['occasional_sum'] = 0  # Якщо немає таких колонок, сума 0
            subtotals_columns.append('occasional_sum')
            self.logger.info("Колонки для 'occasional_sum' відсутні, створено колонку з нулями.")

        return subtotals_columns

    def _find_correlated_aux_features(
            self,
            df_corr_analysis: pd.DataFrame,  # DataFrame для аналізу, вже має містити лише потрібні числові колонки
            target_columns: list[str],
            correlation_threshold: float = 0.5,
            method: str = 'pearson'
    ) -> list[str]:
        """
        Знаходить допоміжні ознаки, які мають кореляцію (за абсолютним значенням)
        вище зазначеного порогу хоча б з однією з цільових колонок.

        Args:
            df_corr_analysis (pd.DataFrame): DataFrame, що містить лише числові колонки для аналізу кореляції.
                                            Має містити як цільові, так і потенційні допоміжні колонки.
            target_columns (list[str]): Список назв цільових колонок.
            correlation_threshold (float, optional): Поріг кореляції. За замовчуванням 0.5.
            method (str, optional): Метод розрахунку кореляції. За замовчуванням 'pearson'.

        Returns:
            list[str]: Відсортований список назв допоміжних колонок, що відповідають умові.
        """
        self.logger.info(f"Пошук корелюючих допоміжних ознак з порогом {correlation_threshold}...")
        if not isinstance(df_corr_analysis, pd.DataFrame):
            self.logger.error("Параметр 'df_corr_analysis' має бути pandas DataFrame.")
            raise ValueError("Параметр 'df_corr_analysis' має бути pandas DataFrame.")
        if not isinstance(target_columns, list) or not all(isinstance(col, str) for col in target_columns):
            self.logger.error("Параметр 'target_columns' має бути списком рядків.")
            raise ValueError("Параметр 'target_columns' має бути списком рядків.")
        if not 0 <= correlation_threshold <= 1:
            self.logger.error("Параметр 'correlation_threshold' має бути в діапазоні [0, 1].")
            raise ValueError("Параметр 'correlation_threshold' має бути в діапазоні [0, 1].")

        # Перевірка наявності цільових стовпців
        valid_target_columns = [col for col in target_columns if col in df_corr_analysis.columns]
        if not valid_target_columns:
            self.logger.warning("Жоден із зазначених цільових стовпців не знайдений у DataFrame для аналізу кореляції.")
            return []

        # Допоміжні стовпці - це всі стовпці в df_corr_analysis, які не є цільовими
        aux_candidate_columns = [col for col in df_corr_analysis.columns if col not in valid_target_columns]

        if not aux_candidate_columns:
            self.logger.warning("Немає допоміжних числових стовпців для аналізу кореляції.")
            return []

        try:
            # Розрахунок матриці кореляції для всіх переданих стовпців
            corr_matrix = df_corr_analysis.corr(method=method)
        except Exception as e:
            self.logger.error(f"Помилка розрахунку матриці кореляції: {e}")
            return []

        # Розглядаємо кореляції допоміжних стовпців з цільовими
        correlations_with_targets = corr_matrix.loc[aux_candidate_columns, valid_target_columns].abs()

        highly_correlated_mask = (correlations_with_targets > correlation_threshold).any(axis=1)
        result_features = correlations_with_targets[highly_correlated_mask].index.tolist()

        self.logger.info(f"Знайдено {len(result_features)} сильно корелюючих допоміжних ознак.")
        return sorted(result_features)

    def process_spending_data(self, file_path: pathlib.Path,
                              correlation_threshold_for_aux_features: float = 0.1) -> tuple[
        pd.DataFrame, dict[str, list[str]], list[str]]:
        """
        Головний метод для обробки даних про витрати.
        Завантажує дані, виконує їх очищення, трансформацію, інженерію ознак
        та повертає оброблений DataFrame, класифікацію колонок та список корелюючих ознак.

        Args:
            file_path (pathlib.Path): Шлях до вхідного CSV файлу з даними про витрати.
            correlation_threshold_for_aux_features (float): Поріг кореляції для функції
                                                            _find_correlated_aux_features.

        Returns:
            tuple[pd.DataFrame, dict[str, list[str]], list[str]]:
                - processed_df (pd.DataFrame): Оброблений DataFrame.
                - classified_categories (dict[str, list[str]]): Словник з категоріями витрат
                  ('regular', 'above_80_nan', 'above_95_nan').
                - correlated_aux_features (list[str]): Список допоміжних ознак, що корелюють
                  з регулярними витратами.

        Приклад виклику:
        ```python
        # from logger_utils import get_logger # Якщо get_logger в окремому файлі
        # logger = get_logger("MainApp")
        # processor = SpendingDataProcessor(logger=logger)
        # df, cat_cols, corr_feats = processor.process_spending_data("D:/PyProj/SpendingPrediction/data/dataset.csv")
        # print(df.head())
        # print(cat_cols)
        # print(corr_feats)
        ```
        """
        self.logger.info(f"Початок обробки даних для файлу: {file_path}")

        # 1. Завантаження та початкова підготовка
        initial_df, all_categories = self._load_and_prepare_initial_df(file_path)
        initial_month_period_nunique = initial_df["month_period"].nunique()  # Для порогів класифікації

        # 2. Створення зведеної таблиці
        # self.df - це основний DataFrame, який буде модифікуватися
        self.df = self._create_pivot_table(initial_df)

        # 3. Класифікація категорій витрат
        classified_categories, regular_cols, periodical_cols, occasional_cols = \
            self._classify_spending_categories(self.df, all_categories, initial_month_period_nunique)

        # 4. Інженерія часових ознак (модифікує self.df)
        self._engineer_temporal_features(self.df)

        # 5. Інженерія агрегованих ознак витрат (модифікує self.df)
        self._engineer_aggregated_spend_features(self.df, periodical_cols, occasional_cols)

        # 6. Пошук корелюючих допоміжних ознак
        # Готуємо DataFrame для аналізу кореляції: лише стовпці витрат
        # Всі стовпці витрат (регулярні, періодичні, рідкісні)
        all_expense_columns_for_corr = [col for col in (regular_cols + periodical_cols + occasional_cols) if
                                        col in self.df.columns]

        if not all_expense_columns_for_corr:
            self.logger.warning(
                "Не знайдено стовпців витрат для аналізу кореляції. Повертаю порожній список корелюючих ознак.")
            correlated_aux_features = []
        else:
            # Використовуємо копію DataFrame лише з потрібними колонками для аналізу
            df_for_correlation_analysis = self.df[all_expense_columns_for_corr].copy()

            # Цільові колонки для пошуку кореляцій - регулярні витрати
            # Переконуємося, що вони є в df_for_correlation_analysis
            valid_regular_cols_for_corr = [col for col in regular_cols if col in df_for_correlation_analysis.columns]

            if not valid_regular_cols_for_corr:
                self.logger.warning(
                    "Цільові регулярні стовпці для кореляції не знайдені в DataFrame. Повертаю порожній список корелюючих ознак.")
                correlated_aux_features = []
            else:
                correlated_aux_features = self._find_correlated_aux_features(
                    df_corr_analysis=df_for_correlation_analysis,
                    target_columns=valid_regular_cols_for_corr,
                    correlation_threshold=correlation_threshold_for_aux_features,
                    method='pearson'
                )

        # 7. Підготовка фінального DataFrame для повернення
        # Видаляємо оригінальні часові колонки та 'month_period'
        cols_to_drop = [col for col in RAW_TEMPORAL_COLUMNS_TO_DROP if col in self.df.columns]
        processed_df = self.df.drop(columns=cols_to_drop)

        self.logger.info("Обробку даних завершено успішно.")
        return processed_df, classified_categories, correlated_aux_features

def main():
    print("Запуск тестового сценарію для SpendingDataProcessor...")

    file_path_data_folder = pathlib.Path(__file__).parents[3].resolve() / 'data'
    file_path = file_path_data_folder / 'dataset.csv'

    processor = DataPreprocessor()

    try:
        final_df, column_categories, correlated_features = processor.process_spending_data(
            file_path)

        if not (file_path_data_folder / 'intermediate').exists():
            pathlib.Path(file_path_data_folder / 'intermediate').mkdir(parents=True)

        pd.to_pickle(final_df, file_path_data_folder / 'intermediate/data_prepare.pkl')

        with open(file_path_data_folder / 'intermediate/columns_cat.json', 'w') as f:
            json.dump(column_categories, f)

        with open(file_path_data_folder / 'intermediate/correlated_features.txt', 'w') as f:
            f.writelines(correlated_features)

        print(f"Розмір DataFrame: {final_df.shape}")

        print(f"\n--- Класифіковані категорії ---")
        for key, val in column_categories.items():
            print(f"{key}: {val}")

        print(f"\n--- Корелюючі допоміжні ознаки (з регулярними витратами, поріг 0.1) ---")
        print(correlated_features)

        # Перевірка наявності очікуваних колонок
        print(f"\n--- Стовпці в фінальному DataFrame ---")
        print(final_df.columns.tolist())


    except Exception as e:
        print(f"Під час тестового запуску сталася помилка: {e}")

if __name__ == '__main__':
    main()