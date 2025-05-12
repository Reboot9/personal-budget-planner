import json
import logging
import pathlib
import warnings
from datetime import date
from typing import Any

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

from predictions.services.main_model.utils import get_logger
from personal_budget_planner.constants import *
import warnings

warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)


class FeatureEngineer:
    """
    Клас для виконання розширеного аналізу даних (EDA) та інженерії ознак
    на основі часових рядів витрат.

    Attributes:
        logger (logging.Logger): Екземпляр логера.
        df (pd.DataFrame): DataFrame, що обробляється.
        features (dict[str, list[str]]): Словник для зберігання груп створених ознак.
        target_cols (list[str]): Список цільових колонок для прогнозування.
        including_secondary_features (list[str]): Список вторинних ознак, які слід включити.
    """

    def __init__(self, logger: logging.Logger | None = None) -> None:
        """
        Ініціалізація інженера ознак.

        Args:
            logger (logging.Logger | None, optional): Екземпляр логера.
                Якщо None, створюється новий логер з ім'ям 'feature_engineer.log'.
                Defaults to None.
        """
        self.logger: logging.Logger = logger if logger else get_logger(self.__class__.__name__,
                                                                       log_file_name='feature_engineer.log')
        self.df: pd.DataFrame = pd.DataFrame()
        self.features: dict[str, list[str]] = {}
        self.target_cols: list[str] = []
        self.including_secondary_features: list[str] = []

        # Ігнорування попереджень для чистішого виводу логів
        warnings.filterwarnings('ignore', category=UserWarning)  # Загальні UserWarning
        warnings.filterwarnings('ignore', category=FutureWarning)  # FutureWarnings від Pandas/Numpy
        warnings.filterwarnings('ignore',
                                message="A value is trying to be set on a copy of a slice from a DataFrame")  # Специфічне попередження PANDAS

    def _check_datetime_index_continuity(self) -> None:
        """
        Перевіряє безперервність часового індексу DataFrame (self.df).

        Логує інформацію про знайдені пропуски. Якщо індекс не є pd.DatetimeIndex,
        логує попередження. Намагається визначити частоту ('D'), якщо вона
        не визначається автоматично, особливо для даних з пропусками.
        """
        if not isinstance(self.df.index, pd.DatetimeIndex):
            self.logger.warning("Індекс DataFrame не є DatetimeIndex. Перевірка безперервності неможлива.")
            return

        self.logger.debug("Перевірка безперервності часового індексу...")
        expected_freq: str | None = pd.infer_freq(self.df.index)

        if self.df.empty:
            self.logger.info("DataFrame порожній, перевірка безперервності індексу не виконується.")
            return

        if not expected_freq:
            # Спроба визначити, чи є він щоденним з пропусками
            is_daily_normalized: bool = all(self.df.index == self.df.index.normalize())
            diffs: pd.Series = self.df.index.to_series().diff().dropna()
            # Якщо всі інтервали - 1 день або індекс нормалізований, але infer_freq не спрацював через пропуски
            if not diffs.empty and (diffs == pd.Timedelta(days=1)).all():
                expected_freq = 'D'
            elif is_daily_normalized and diffs.empty and len(self.df.index) == 1:  # Один запис
                expected_freq = 'D'  # Припускаємо щоденну частоту для одного запису, якщо він нормалізований
            elif is_daily_normalized and not diffs.empty:  # Кілька записів, нормалізовані, але infer_freq не спрацював
                self.logger.warning(
                    "Не вдалося автоматично визначити частоту індексу (можливо, 'D' з пропусками). Припускаємо 'D' для перевірки.")
                expected_freq = 'D'
            else:
                self.logger.warning(
                    f"Не вдалося визначити очікувану частоту індексу. Поточний inferred_freq: {expected_freq}. Пропускаємо детальну перевірку на пропуски.")
                self.logger.info(f"Поточний діапазон індексу: від {self.df.index.min()} до {self.df.index.max()}")
                return

        try:
            actual_range: pd.DatetimeIndex = pd.date_range(start=self.df.index.min(), end=self.df.index.max(), freq=expected_freq)
            missing: pd.DatetimeIndex = actual_range.difference(self.df.index)
            if not missing.empty:
                self.logger.warning(
                    f"Виявлено {len(missing)} пропущених періодів ({expected_freq}) в індексі! Перші 5: {missing[:5].to_list()}")
            else:
                self.logger.info(f"Часовий індекс є безперервним з очікуваною частотою {expected_freq}.")
        except Exception as e:
            self.logger.error(f"Помилка під час перевірки безперервності індексу з частотою {expected_freq}: {e}")

    def _check_features(self, verbose: bool = False) -> list[str]:
        """
        Перевіряє наявність ознак, описаних у словнику self.features, у self.df.

        Логує інформацію про ознаки зі словника `self.features`, які відсутні в DataFrame `self.df`.

        Args:
            verbose (bool, optional): Якщо True, логує повідомлення про успішну перевірку,
                                      коли всі ознаки присутні. Defaults to False.

        Returns:
            list[str]: Список назв ознак, які є в значеннях словника `self.features`,
                       але відсутні в колонках `self.df`.
        """
        self.logger.debug("Запуск _check_features.")
        all_features_in_dict_values: list[str] = []
        for key, feature_list_val in self.features.items():
            if isinstance(feature_list_val, list):
                all_features_in_dict_values.extend(feature_list_val)
            elif isinstance(feature_list_val, str):  # На випадок, якщо значення не список
                all_features_in_dict_values.append(feature_list_val)

        unique_features_in_dict: set[str] = set(all_features_in_dict_values)
        df_columns: set[str] = set(self.df.columns)

        missing_in_df: list[str] = list(unique_features_in_dict - df_columns)

        if missing_in_df:
            self.logger.warning(
                f"Ознаки, перераховані в словнику `features`, але ВІДСУТНІ в DataFrame: {missing_in_df}")
        elif verbose:
            self.logger.info("Усі ознаки зі словника `features` присутні в DataFrame.")

        return missing_in_df

    def _aggregate_spending_categories(self, irregular_cols: list[str], non_periodic_cols: list[str]) -> None:
        """
        Агрегує вказані колонки витрат у тематичні групи та обчислює загальну суму для цих груп.

        Модифікує `self.df` на місці, додаючи колонки 'IrregularSpendings',
        'NonPeriodicSpendings' та 'AllListedSpendings'.

        Args:
            irregular_cols (list[str]): Список назв колонок, що представляють нерегулярні витрати.
            non_periodic_cols (list[str]): Список назв колонок, що представляють неперіодичні витрати.
        """
        self.logger.info("Агрегація категорій витрат...")

        if irregular_cols:
            self.df['IrregularSpendings'] = self.df[irregular_cols].sum(axis=1, min_count=0)
        else:
            self.df['IrregularSpendings'] = 0.0

        if non_periodic_cols:
            self.df['NonPeriodicSpendings'] = self.df[non_periodic_cols].sum(axis=1, min_count=0)
        else:
            self.df['NonPeriodicSpendings'] = 0.0

        # target_cols встановлюються в public-методі
        all_listed_cols_for_sum: list[str] = list(set(self.target_cols + irregular_cols + non_periodic_cols))
        # Переконуємося, що ці колонки існують в self.df
        all_listed_cols_for_sum = [col for col in all_listed_cols_for_sum if col in self.df.columns]

        if all_listed_cols_for_sum:
            self.df['AllListedSpendings'] = self.df[all_listed_cols_for_sum].sum(axis=1, min_count=0)
        else:
            self.df['AllListedSpendings'] = 0.0
        self.logger.info(
            "Агреговані колонки 'IrregularSpendings', 'NonPeriodicSpendings', 'AllListedSpendings' створено/оновлено.")

    def _fill_missing_dates(self) -> None:
        """
        Заповнює пропущені дати в індексі DataFrame `self.df` нулями.

        Метод сортує DataFrame за індексом, визначає повний діапазон дат
        з частотою 'D' (щоденно) від мінімальної до максимальної дати в індексі,
        знаходить відсутні дати та додає для них рядки, заповнені нулями.
        Якщо DataFrame порожній, індекс не є pd.DatetimeIndex, або містить
        менше двох записів, заповнення не виконується.
        """
        self.logger.info("Заповнення пропущених дат в індексі...")
        self.df = self.df.sort_index()
        if self.df.empty or not isinstance(self.df.index, pd.DatetimeIndex) or len(
                self.df.index) < 1:  # len(self.df.index) < 2 було додано для date_range, але <1 достатньо для перевірки порожнечі
            self.logger.info("DataFrame порожній або індекс не підходить для заповнення дат.")
            return

        # Для pd.date_range потрібен хоча б один елемент, а для визначення freq - хоча б два.
        # Якщо тільки один елемент, ми не можемо визначити freq чи заповнити проміжки, але можемо встановити freq='D' явно.
        if len(self.df.index) < 2: # Оригінальний коментар: "Якщо тільки один елемент, ми не можемо визначити freq чи заповнити проміжки."
                                   # Однак, для date_range з явною частотою 'D' це не проблема.
            self.logger.info("Недостатньо даних в індексі для автоматичного визначення діапазону та частоти для заповнення пропусків, але спробуємо з freq='D'.")
            # Навіть з одним елементом, якщо ми хочемо заповнити до нього або після нього, це можливо, але тут логіка заповнює *між* min і max.
            # Якщо тільки один запис, min() == max(), тому full_range буде містити тільки цю одну дату, і missing_dates буде порожнім.
            # Це коректно.

        try:
            full_range: pd.DatetimeIndex = pd.date_range(start=self.df.index.min(), end=self.df.index.max(), freq='D')
            missing_dates: pd.DatetimeIndex = full_range.difference(self.df.index)
            if not missing_dates.empty:
                self.logger.info(f"Знайдено {len(missing_dates)} пропущених дат. Заповнення нулями...")
                zero_rows: pd.DataFrame = pd.DataFrame(0, index=missing_dates, columns=self.df.columns)
                self.df = pd.concat([self.df, zero_rows]).sort_index()
                self.logger.info(f"Розмір таблиці після заповнення пропущених дат: {self.df.shape}")
            else:
                self.logger.info("Пропущених дат не знайдено.")
        except Exception as e:
            self.logger.error(f"Помилка під час заповнення пропущених дат: {e}")

    def _apply_log_transformation(self) -> None:
        """
        Застосовує логарифмічну трансформацію log(1+x) до всіх числових колонок витрат у `self.df`.

        Створює нові колонки з суфіксом '_log'. Назви нових колонок додаються
        до `self.features['log_columns']`.
        """
        self.logger.info("Застосування логарифмічної трансформації (log1p)...")
        log_cols: list[str] = []
        # Використовуємо поточні колонки self.df, які є числовими і не є вже логарифмованими
        cols_for_log: list[str] = [col for col in self.df.columns
                                   if pd.api.types.is_numeric_dtype(self.df[col]) and not col.endswith('_log')]

        for col in cols_for_log:
            log_col_name = f'{col}_log'
            self.df[log_col_name] = np.log1p(self.df[col])
            log_cols.append(log_col_name)

        if 'log_columns' not in self.features:
            self.features['log_columns'] = []
        self.features['log_columns'].extend(log_cols)
        self.features['log_columns'] = sorted(list(set(self.features['log_columns'])))  # Унікальні та відсортовані
        self.logger.info(f"Створено {len(log_cols)} логарифмованих ознак.")

    def _check_stationarity(self) -> None:
        """
        Перевіряє стаціонарність часових рядів за допомогою тесту Дікі-Фуллера (ADF).

        Тест проводиться для цільових колонок (`self.target_cols`) та їхніх
        логарифмованих версій (якщо вони існують у `self.df`).
        Результати (p-value та висновок про стаціонарність) логуються.
        Нульова гіпотеза (H0): ряд нестаціонарний. Альтернативна (H1): ряд стаціонарний.
        Якщо p-value < 0.05, H0 відхиляється.
        """
        self.logger.info("Перевірка стаціонарності для цільових та логарифмованих цільових категорій (ADF тест)...")
        self.logger.info("H0: Ряд нестаціонарний. H1: Ряд стаціонарний. Якщо p-value < 0.05, відхиляємо H0.")

        # Колонки для тестування - цільові та їх логарифмовані версії
        cols_to_test_adf: list[str] = self.target_cols + [f"{col}_log" for col in self.target_cols if
                                                          f"{col}_log" in self.df.columns]

        for col_name in cols_to_test_adf:
            if col_name not in self.df.columns:
                self.logger.warning(f"Колонка {col_name} для ADF тесту відсутня.")
                continue
            series_to_test: pd.Series = self.df[col_name].dropna()
            if series_to_test.empty or len(series_to_test) < 10:  # ADF потребує достатньо даних
                self.logger.info(
                    f"Колонка '{col_name}': недостатньо даних для ADF тесту ({len(series_to_test)} точок).")
                continue

            try:
                result: tuple[float, float, int, int, dict[str, float], float] = adfuller(series_to_test)
                p_value: float = result[1]
                if p_value < 0.05:
                    self.logger.info(f"  Ряд '{col_name}', ймовірно, стаціонарний (p-value: {p_value:.4f}).")
                else:
                    self.logger.info(
                        f"  Ряд '{col_name}', ймовірно, НЕстаціонарний (p-value: {p_value:.4f}). ADF Stat: {result[0]:.4f}.")
            except Exception as e:
                self.logger.error(f"Помилка при виконанні ADF тесту для колонки '{col_name}': {e}")

    @staticmethod
    def _week_of_month(dt: date) -> int | None:
        """
        Повертає номер календарного тижня в місяці (починаючи з 0) для заданої дати.

        Розрахунок враховує випадки, коли тиждень на початку або в кінці року
        може належати іншому року за ISO календарем.
        Примітка: Логіка може повертати None у деяких випадках, коли week_in_month >= 0
        без входження в умову week_in_month < 0.

        Args:
            dt (date): Дата, для якої розраховується номер тижня в місяці.

        Returns:
            int | None: Номер тижня в місяці (0-індексований) або None,
                        якщо розрахунок не призводить до цілого числа за поточною логікою.
        """
        first_day: date = dt.replace(day=1)
        first_calendar_week_of_month: int = first_day.isocalendar().week
        current_calendar_week: int = dt.isocalendar().week

        if dt.month == 1 and first_calendar_week_of_month > 50:  # Січень, перший день на тижні 52/53 попереднього року
            first_calendar_week_of_month = 0  # фактично робить його тижнем 0 для розрахунку
        elif dt.month == 12 and current_calendar_week == 1:  # Грудень, дата на тижні 1 наступного року
            current_calendar_week = first_day.isocalendar().week + (dt - first_day).days // 7 + 1  # приблизно

        week_in_month: int = current_calendar_week - first_calendar_week_of_month
        if week_in_month < 0:
            offset: int = 0
            if dt.isocalendar()[0] > first_day.isocalendar()[0]:  # Поточна дата належить наступному ISO року
                offset = first_day.replace(month=12, day=31).isocalendar().week  # тижнів у році first_day
                if first_day.isocalendar().week > current_calendar_week:  # напр., перший день - тиждень 53, поточний - тиждень 1 наступного року
                    offset = first_day.replace(month=12, day=31).isocalendar().week - first_day.isocalendar().week + 1
                else:  # поточна дата - пізніший тиждень наступного року
                    offset = first_day.replace(month=12, day=31).isocalendar().week

            week_in_month = (current_calendar_week + offset) - first_calendar_week_of_month
            return week_in_month
        # Відповідно до оригінальної логіки, якщо week_in_month не було < 0, повертається None.
        return None


    def _create_calendar_features(self) -> None:
        """
        Створює календарні ознаки на основі індексу DataFrame `self.df`.

        Додає такі колонки: 'day_of_year', 'week_of_year', 'quarter', 'year',
        'month', 'day_of_week', 'day_of_month', 'week_of_month', 'is_weekend',
        'is_month_start', 'is_month_end'.
        Назви створених ознак додаються до `self.features['calendar_features']`.
        Вимагає, щоб індекс `self.df` був pd.DatetimeIndex.
        """
        self.logger.info("Створення календарних ознак...")
        if not isinstance(self.df.index, pd.DatetimeIndex):
            self.logger.error("Неможливо створити календарні ознаки: індекс не є DatetimeIndex.")
            return

        self.df['day_of_year'] = self.df.index.dayofyear
        self.df['week_of_year'] = self.df.index.isocalendar().week.astype(int)
        self.df['quarter'] = self.df.index.quarter
        self.df['year'] = self.df.index.year  # Рік вже може існувати з попереднього етапу
        self.df['month'] = self.df.index.month
        self.df['day_of_week'] = self.df.index.dayofweek  # Понеділок=0, Неділя=6
        self.df['day_of_month'] = self.df.index.day

        # Застосування _week_of_month, якщо індекс не порожній
        if not self.df.index.empty:
            self.df['week_of_month'] = [self._week_of_month(idx.date()) for idx in self.df.index]
        else:
            self.df['week_of_month'] = pd.Series(dtype='object') # Змінено dtype на 'object' для узгодження з можливим None

        self.df['is_weekend'] = self.df['day_of_week'].isin([5, 6]).astype(int)
        self.df['is_month_start'] = self.df['day_of_month'].isin([1, 2, 3]).astype(int)
        self.df['is_month_end'] = (self.df.index.is_month_end).astype(int)

        calendar_features: list[str] = ['day_of_week', 'day_of_month', 'day_of_year',
                                        'week_of_year', 'week_of_month', 'month', 'quarter', 'year',
                                        'is_weekend', 'is_month_start', 'is_month_end']
        self.features['calendar_features'] = sorted(list(set(calendar_features)))
        self.logger.info(f"Створено календарні ознаки: {calendar_features}")

    def _create_cyclical_features(self) -> None:
        """
        Створює циклічні ознаки (sin/cos трансформації) для часових компонентів.

        Трансформує 'day_of_week', 'month', 'day_of_year' (якщо вони існують у `self.df`)
        у їхні sin та cos компоненти для кращого представлення циклічності в моделях.
        Назви створених ознак додаються до `self.features['cyclical_features']`.
        """
        self.logger.info("Створення циклічних часових ознак...")
        # День тижня
        if 'day_of_week' in self.df.columns:
            self.df['day_of_week_sin'] = np.sin(2 * np.pi * self.df['day_of_week'] / 7)
            self.df['day_of_week_cos'] = np.cos(2 * np.pi * self.df['day_of_week'] / 7)
        # Місяць
        if 'month' in self.df.columns:
            self.df['month_sin'] = np.sin(2 * np.pi * self.df['month'] / 12)
            self.df['month_cos'] = np.cos(2 * np.pi * self.df['month'] / 12)
        # День року
        if 'day_of_year' in self.df.columns:
            # Визначення, чи є рік високосним для правильного ділення для day_of_year
            days_in_year: np.ndarray = np.where(self.df.index.is_leap_year, 366, 365)
            self.df['day_of_year_sin'] = np.sin(2 * np.pi * self.df['day_of_year'] / days_in_year)
            self.df['day_of_year_cos'] = np.cos(2 * np.pi * self.df['day_of_year'] / days_in_year)

        cyclical_features: list[str] = [col for col in self.df.columns if '_sin' in col or '_cos' in col]
        self.features['cyclical_features'] = sorted(list(set(cyclical_features)))
        self.logger.info(f"Створено циклічні ознаки: {self.features['cyclical_features']}")

    def _create_lag_features(self) -> None:
        """
        Створює лагові ознаки для цільових колонок (`self.target_cols`).

        Для кожної цільової колонки створюються нові колонки зі значеннями,
        зсунутими на кількість періодів, вказаних у константі `LAGS`.
        Рядки з NaN, що виникають через зсув, видаляються.
        Назви створених лагових ознак додаються до `self.features['lag_features']`.
        """
        self.logger.info(f"Створення лагових ознак для {self.target_cols} з лагами: {LAGS}...")
        lag_cols: list[str] = []
        for col in self.target_cols:
            if col not in self.df.columns:
                self.logger.warning(f"Цільова колонка '{col}' для створення лагів відсутня в DataFrame.")
                continue
            for lag in LAGS:
                lag_col_name: str = f'{col}_lag_{lag}'
                self.df[lag_col_name] = self.df[col].shift(lag)
                lag_cols.append(lag_col_name)

        if lag_cols:  # Тільки якщо лаги були створені
            initial_rows: int = self.df.shape[0]
            self.df.dropna(subset=lag_cols, inplace=True)  # Викидаємо рядки з NaN, створені лагами
            rows_dropped: int = initial_rows - self.df.shape[0]
            self.logger.info(
                f"Видалено {rows_dropped} рядків через NaN у лагових ознаках (макс. лаг: {max(LAGS)}).")

        self.features['lag_features'] = sorted(list(set(lag_cols)))
        self.logger.info(f"Створено {len(lag_cols)} лагових ознак.")

    @staticmethod
    def _rolling_mad(x: np.ndarray) -> float:
        """
        Обчислює медіанне абсолютне відхилення (MAD) для ковзного вікна.

        Args:
            x (np.ndarray): Масив значень, для яких обчислюється MAD.

        Returns:
            float: Значення MAD. Повертає np.nan, якщо вхідний масив порожній
                   або всі значення є NaN.
        """
        if len(x) == 0 or np.all(np.isnan(x)):
            return np.nan
        median: float = np.nanmedian(x)
        return np.nanmedian(np.abs(x - median))

    @staticmethod
    def _rolling_trimmed_mean(x: np.ndarray, proportiontocut: float = 0.1) -> float:
        """
        Обчислює усічене середнє для ковзного вікна.

        Видаляє вказану частку (`proportiontocut`) найменших та найбільших
        значень перед обчисленням середнього. NaN значення видаляються перед усіченням.

        Args:
            x (np.ndarray): Масив значень, для яких обчислюється усічене середнє.
            proportiontocut (float, optional): Частка значень, що видаляється з кожного
                                               кінця відсортованого масиву. Defaults to 0.1.

        Returns:
            float: Усічене середнє. Повертає np.nan, якщо після видалення NaN
                   масив порожній. Якщо після усічення не лишається значень,
                   повертає медіану вихідного очищеного масиву.
        """
        x_clean: np.ndarray = x[~np.isnan(x)]  # Видаляємо NaN перед обробкою
        if len(x_clean) == 0:
            return np.nan
        n: int = len(x_clean)
        lowercut: int = int(n * proportiontocut)
        uppercut: int = n - lowercut
        if lowercut >= uppercut:  # Якщо після усічення нічого не лишається
            return np.nanmedian(x_clean)  # Повертаємо медіану як робастну оцінку
        x_sorted: np.ndarray = np.sort(x_clean)
        trimmed_x: np.ndarray = x_sorted[lowercut:uppercut]
        return np.mean(trimmed_x) if len(trimmed_x) > 0 else np.nan

    def _create_rolling_window_features(self) -> None:
        """
        Створює ознаки ковзного вікна (стандартні та робастні) для цільових колонок.

        Для кожної цільової колонки (`self.target_cols`) та для кожного розміру вікна
        з константи `WINDOWS` обчислюються різні статистичні показники
        (середнє, стандартне відхилення, сума, медіана, асиметрія, ексцес,
        квантилі, IQR, MAD, усічене середнє).
        Використовується зсув на 1 (`shift(1)`) для уникнення витоку даних.
        NaN значення, що виникають на початку ряду, заповнюються нулями.
        Назви створених ознак додаються до `self.features['rolling_cols']`.
        """
        self.logger.info(f"Створення ознак ковзного вікна для {self.target_cols} з вікнами: {WINDOWS}...")
        rolling_cols: list[str] = []

        for col in self.target_cols:
            if col not in self.df.columns:
                self.logger.warning(f"Цільова колонка '{col}' для ковзних вікон відсутня.")
                continue
            for window in WINDOWS:
                shifted_series: pd.Series = self.df[col].shift(1)  # shift(1) для уникнення витоку даних

                # Стандартні статистики
                ops: dict[str, Any] = {'mean': np.mean, 'std': np.std, 'sum': np.sum,
                       'median': np.median, 'skew': pd.Series.skew, 'kurt': pd.Series.kurt,
                       'q25': lambda x: np.percentile(x, 25) if not np.all(np.isnan(x)) and len(
                           x[~np.isnan(x)]) > 0 else np.nan,  # Робота з квантилями
                       'q75': lambda x: np.percentile(x, 75) if not np.all(np.isnan(x)) and len(
                           x[~np.isnan(x)]) > 0 else np.nan,
                       }
                for op_name, op_func in ops.items():
                    r_col_name: str = f'{col}_roll_{op_name}_{window}'
                    # .apply(op_func, raw=True if op_name in ['mean', 'std', 'sum', 'median'] else False) # raw=True для numpy функцій
                    if op_name in ['skew', 'kurt']:  # Ці методи є в Series.rolling
                        self.df[r_col_name] = shifted_series.rolling(window=window, closed='left').agg(op_name)
                    else:  # Для інших використовуємо apply або вбудовані агрегатори
                        self.df[r_col_name] = shifted_series.rolling(window=window, closed='left').agg(
                            op_func if callable(op_func) else op_name)
                    rolling_cols.append(r_col_name)

                # IQR (Міжквартильний розмах)
                q75_col: str = f'{col}_roll_q75_{window}'
                q25_col: str = f'{col}_roll_q25_{window}'
                if q75_col in self.df.columns and q25_col in self.df.columns:
                    self.df[f'{col}_roll_iqr_{window}'] = self.df[q75_col] - self.df[q25_col]
                    rolling_cols.append(f'{col}_roll_iqr_{window}')

                # Кастомні робастні статистики
                self.df[f'{col}_roll_mad_{window}'] = shifted_series.rolling(window=window, closed='left').apply(
                    self._rolling_mad, raw=True)
                rolling_cols.append(f'{col}_roll_mad_{window}')
                self.df[f'{col}_roll_trimmean_{window}'] = shifted_series.rolling(window=window, closed='left').apply(
                    lambda x: self._rolling_trimmed_mean(x, 0.1), raw=True)
                rolling_cols.append(f'{col}_roll_trimmean_{window}')

        # Заповнення NaN, що виникли через ковзні вікна (особливо на початку ряду)
        self.df.fillna(0, inplace=True)
        self.logger.info(f"Загальна кількість створених ознак ковзного вікна: {len(rolling_cols)}")
        self.features['rolling_cols'] = sorted(list(set(rolling_cols)))

    def _create_time_since_and_frequency_features(self) -> None:
        """
        Створює ознаки часу з останньої витрати та частоти витрат для цільових колонок.

        Для кожної цільової колонки (`self.target_cols`):
        1.  'time_since_last': Обчислює кількість днів з моменту останньої ненульової витрати.
        2.  'freq_{window}': Обчислює кількість ненульових витрат у ковзних вікнах
            розміром 30, 60, 90 днів (з зсувом на 1).
        NaN значення, що виникають, заповнюються нулями.
        Назви створених ознак додаються до `self.features['times_since_cols']`
        та `self.features['freq_cols']`.
        """
        self.logger.info(f"Створення ознак часу з останньої витрати та частоти для {self.target_cols}...")
        time_since_cols: list[str] = []
        freq_cols: list[str] = []

        for col in self.target_cols:
            if col not in self.df.columns:
                self.logger.warning(f"Цільова колонка '{col}' для ознак часу/частоти відсутня.")
                continue
            # Час з останньої ненульової витрати
            time_since_col: str = f'{col}_time_since_last'
            spending_days: pd.Series = self.df[col] > 0
            cum_days: pd.Series = (~spending_days).cumsum()
            # ffill() заповнює NaN значеннями, які йдуть перед ними в кумулятивній сумі скинутих лічильників
            self.df[time_since_col] = cum_days - cum_days.where(spending_days).ffill().fillna(0)
            time_since_cols.append(time_since_col)

            # Частота: Кількість ненульових витрат у ковзних вікнах
            for window in [30, 60, 90]: # Розміри вікон для частоти
                freq_col_name: str = f'{col}_freq_{window}'
                self.df[freq_col_name] = (self.df[col] > 0).shift(1).rolling(window=window, closed='left').sum().fillna(
                    0)
                freq_cols.append(freq_col_name)

        self.df.fillna(0, inplace=True)  # Заповнення NaN від rolling для freq
        self.features['times_since_cols'] = sorted(list(set(time_since_cols)))
        self.features['freq_cols'] = sorted(list(set(freq_cols)))
        self.logger.info(
            f"Створено {len(time_since_cols)} ознак 'час з останньої витрати' та {len(freq_cols)} ознак 'частота витрат'.")

    def _create_domain_specific_features(self, irregular_spend_col: str = 'IrregularSpendings',
                                         non_periodic_spend_col: str = 'NonPeriodicSpendings') -> None:
        """
        Створює доменно-специфічні та нестандартні ознаки.

        1.  Співвідношення витрат: Розраховує частку витрат кожної цільової категорії
            відносно загальних витрат ('AllListedSpendings') за ковзне вікно `WINDOW_RATIO`.
        2.  Вплив великих витрат: Створює бінарну ознаку 'large_spend_yesterday',
            яка вказує, чи були великі (95-й перцентиль) нецільові витрати
            (з `irregular_spend_col` або `non_periodic_spend_col`) попереднього дня.
            Також обчислює суму нецільових витрат за ковзне вікно `WINDOW_NON_TARGET`.
        NaN значення заповнюються нулями.
        Назви створених ознак додаються до `self.features['domain_features']`.

        Args:
            irregular_spend_col (str, optional): Назва колонки з нерегулярними витратами.
                                                 Defaults to 'IrregularSpendings'.
            non_periodic_spend_col (str, optional): Назва колонки з неперіодичними витратами.
                                                    Defaults to 'NonPeriodicSpendings'.
        """
        self.logger.info("Створення доменно-специфічних ознак...")
        domain_features: list[str] = []

        # 1. Співвідношення витрат (витрати цільової категорії / загальні витрати)
        # Використовуємо 'AllListedSpendings' як загальні витрати
        total_spend_col_rolling: str = 'AllListedSpendings_roll_sum_30'  # Тимчасова колонка
        if 'AllListedSpendings' in self.df.columns:
            self.df[total_spend_col_rolling] = self.df['AllListedSpendings'].shift(1).rolling(window=WINDOW_RATIO,
                                                                                              closed='left').sum().fillna(0)
        else:
            self.df[total_spend_col_rolling] = 0  # Якщо немає загальних, то і співвідношення 0
            self.logger.warning("Колонка 'AllListedSpendings' відсутня для розрахунку співвідношень.")

        ratio_cols_created: list[str] = []
        for col in self.target_cols:
            if col not in self.df.columns:
                self.logger.warning(f"Цільова колонка '{col}' для співвідношень відсутня.")
                continue
            ratio_col_name: str = f'{col}_ratio_{WINDOW_RATIO}d'
            # Має бути створена в _create_rolling_window_features
            category_rolling_sum_col: str = f'{col}_roll_sum_{WINDOW_RATIO}'

            if category_rolling_sum_col in self.df.columns and total_spend_col_rolling in self.df.columns:
                self.df[ratio_col_name] = self.df[category_rolling_sum_col].divide(
                    self.df[total_spend_col_rolling] + 1e-9)  # +1e-9 для уникнення ділення на 0
                self.df[ratio_col_name].fillna(0, inplace=True)
                self.df[ratio_col_name].replace([np.inf, -np.inf], 0, inplace=True)
                ratio_cols_created.append(ratio_col_name)
            else:
                self.logger.warning(
                    f"Необхідні колонки для розрахунку співвідношення {ratio_col_name} відсутні ({category_rolling_sum_col} або {total_spend_col_rolling}).")
        domain_features.extend(ratio_cols_created)
        if total_spend_col_rolling in self.df.columns:  # Видаляємо тимчасову колонку
            self.df.drop(columns=[total_spend_col_rolling], inplace=True)

        # 2. Вплив великих витрат (з нецільових категорій)
        non_target_category_cols: list[str] = []
        if irregular_spend_col in self.df.columns: non_target_category_cols.append(irregular_spend_col)
        if non_periodic_spend_col in self.df.columns: non_target_category_cols.append(non_periodic_spend_col)

        if non_target_category_cols:
            # Розгортання всіх ненульових витрат з цих нецільових категорій
            all_non_target_spends_flat: pd.Series = self.df[non_target_category_cols].unstack().replace(0, np.nan).dropna()
            large_spend_threshold: float
            if not all_non_target_spends_flat.empty:
                large_spend_threshold = all_non_target_spends_flat.quantile(0.95)
            else:  # Якщо немає нецільових витрат взагалі
                large_spend_threshold = np.inf  # Поріг, який ніколи не буде досягнуто
            self.logger.info(f"  Поріг для 'великих нецільових витрат' (95-й перцентиль): {large_spend_threshold:.2f}")

            # Ознаки на основі порогу
            self.df['large_spend_yesterday'] = (self.df[non_target_category_cols].shift(1) > large_spend_threshold).any(
                axis=1).astype(int)
            domain_features.append('large_spend_yesterday')

            # Сума нецільових витрат за останні N днів
            non_target_sum_col: str = f'non_target_spend_sum_{WINDOW_NON_TARGET}d'
            self.df[non_target_sum_col] = self.df[non_target_category_cols].shift(1).rolling(
                window=WINDOW_NON_TARGET, closed='left').sum().sum(axis=1).fillna(0)
            domain_features.append(non_target_sum_col)
        else:
            self.logger.warning("Немає нецільових колонок для розрахунку впливу великих витрат.")
            self.df['large_spend_yesterday'] = 0  # Створюємо, щоб уникнути помилок
            domain_features.append('large_spend_yesterday')
            self.df[f'non_target_spend_sum_{WINDOW_NON_TARGET}d'] = 0
            domain_features.append(f'non_target_spend_sum_{WINDOW_NON_TARGET}d')

        self.df.fillna(0, inplace=True)
        self.features['domain_features'] = sorted(list(set(domain_features)))
        self.logger.info(f"Завершено створення доменних ознак. Створено: {self.features['domain_features']}")

    def engineer_features(self,
                          input_df: pd.DataFrame,
                          column_categories: dict[str, list[str]],
                          target_cols: list[str],
                          including_secondary_features: list[str] | None = None
                          ) -> tuple[pd.DataFrame, dict[str, list[str]]]:
        """
        Головний метод для виконання інженерії ознак.

        Обробляє вхідний DataFrame, створюючи різноманітні ознаки: агреговані,
        логарифмовані, календарні, циклічні, лагові, ковзного вікна,
        часу з останньої події, частоти та доменно-специфічні.
        Заповнює пропущені дати та обробляє NaN значення.

        Args:
            input_df (pd.DataFrame): Вхідний DataFrame. Очікується, що він має колонку 'date'
                                     або вже встановлений pd.DatetimeIndex.
                                     (результат попередньої обробки, наприклад, pivot_df.pkl).
            column_categories (dict[str, list[str]]): Словник з категоріями колонок
                                                      (наприклад, "regular", "above_80_nan", "above_95_nan").
            target_cols (list[str]): Список цільових колонок для прогнозування.
            including_secondary_features (list[str] | None, optional): Список вторинних ознак,
                                                                       які слід явно включити,
                                                                       якщо `DROP_SECONDARY_FEATURES` увімкнено.
                                                                       Defaults to None.

        Returns:
            tuple[pd.DataFrame, dict[str, list[str]]]:
                - pd.DataFrame: DataFrame з новими ознаками.
                - dict[str, list[str]]: Словник, що групує назви створених ознак за категоріями.

        Raises:
            KeyError: Якщо колонка 'date' для індексу не знайдена у вхідному DataFrame
                      і індекс ще не є DatetimeIndex.
            Exception: Якщо не вдається встановити DatetimeIndex.

        Приклад виклику:
        ```python
        # from app.predictor.utils import get_logger # Потрібно для прикладу
        # logger = get_logger("FeatureEngineeringApp")
        # engineer = FeatureEngineer(logger=logger)
        # # Припустимо, `processed_df` та `categories_dict` завантажені або отримані
        # # `targets` - це список цільових колонок, наприклад ['Market', 'Coffee']
        # final_features_df, features_dictionary = engineer.engineer_features(
        #     input_df=processed_df,
        #     column_categories=categories_dict,
        #     target_cols=targets,
        #     including_secondary_features=['SomeSecondaryFeature1']
        # )
        # logger.info(f"Фінальний DataFrame: {final_features_df.shape}")
        # logger.info(f"Словник ознак: {list(features_dictionary.keys())}")
        ```
        """
        self.logger.info("Початок процесу інженерії ознак...")
        self.df = input_df.copy()
        self.target_cols = target_cols
        if including_secondary_features is not None:
            self.including_secondary_features = including_secondary_features

        if not isinstance(self.df.index, pd.DatetimeIndex):
            try:
                self.df['date'] = pd.to_datetime(self.df['date'])
                self.df.set_index('date', inplace=True)
                self.logger.info("Встановлено індекс 'date' як DatetimeIndex.")
            except KeyError:
                self.logger.error("Колонка 'date' для індексу не знайдена у вхідному DataFrame.")
                raise
            except Exception as e:
                self.logger.error(f"Не вдалося встановити DatetimeIndex: {e}")
                raise

        # Отримання списків колонок з категорій
        regular_cols: list[str] = column_categories.get("regular", [])
        irregular_cols: list[str] = column_categories.get("above_80_nan", [])  # У Jupyter Notebook це "irregular_cols"
        non_periodic_cols: list[str] = column_categories.get("above_95_nan", [])  # У Jupyter Notebook це "non_periodic_cols"

        self.logger.info(f"Регулярні витрати: {regular_cols}")
        self.logger.info(f"Нерегулярні витрати (80%+ NaN): {irregular_cols}")
        self.logger.info(f"Неперіодичні витрати (95%+ NaN): {non_periodic_cols}")
        self.logger.info(f"Цільові колонки для прогнозування: {self.target_cols}")

        # 1. Агрегація категорій витрат
        self._aggregate_spending_categories(irregular_cols, non_periodic_cols)

        # Фільтрація колонок згідно DROP_SECONDARY_FEATURES
        if DROP_SECONDARY_FEATURES:
            subtotal_cols: list[str] = ['IrregularSpendings', 'NonPeriodicSpendings', 'AllListedSpendings']
            # Колонки, які точно залишаються: цільові + дозволені вторинні + агрегати
            cols_to_keep: list[str] = self.target_cols + \
                                      [col for col in (non_periodic_cols + irregular_cols) if
                                       col in self.including_secondary_features] + \
                                      subtotal_cols
            # Видаляємо дублікати та переконуємось, що колонки існують
            cols_to_keep = sorted(list(set(c for c in cols_to_keep if c in self.df.columns)))
            self.df = self.df[cols_to_keep].copy()
            self.logger.info(f"Залишено колонки після DROP_SECONDARY_FEATURES: {cols_to_keep}")

        # 2. Ініціалізація словника `features`
        aggregations_in_df: list[str] = [col for col in ['IrregularSpendings', 'NonPeriodicSpendings', 'AllListedSpendings'] if
                                         col in self.df.columns]
        other_non_aggregated_features_in_df: list[str] = [col for col in self.df.columns if
                                                          col not in self.target_cols and col not in aggregations_in_df]

        self.features = {
            'target': [col for col in self.target_cols if col in self.df.columns],  # Тільки ті, що є в df
            'other_non_aggregated_features': other_non_aggregated_features_in_df,
            'aggregated_features': aggregations_in_df
        }
        self.logger.info(
            f"Початкові групи ознак: target({len(self.features['target'])}), other_non_aggregated({len(self.features['other_non_aggregated_features'])}), aggregated({len(self.features['aggregated_features'])}).")

        # 3. Перевірка індексу та заповнення пропущених дат
        self._check_datetime_index_continuity()
        self._fill_missing_dates()
        self._check_datetime_index_continuity()  # Повторна перевірка

        # 4. Логарифмічна трансформація
        self._apply_log_transformation()
        self._check_features(verbose=True)

        # 5. Перевірка стаціонарності (аналітичний крок, не змінює df)
        self._check_stationarity()

        # 6. Створення календарних ознак
        self._create_calendar_features()
        self._check_features()

        # 7. Створення циклічних ознак
        self._create_cyclical_features()
        self._check_features()

        # 8. Створення лагових ознак
        self._create_lag_features()
        self._check_features()
        self._check_datetime_index_continuity()  # Після dropna через лаги

        # 9. Створення ознак ковзного вікна
        self._create_rolling_window_features()
        self._check_features(verbose=True)
        self._check_datetime_index_continuity()  # Після fillna від ковзних вікон

        # 10. Створення ознак часу з останньої витрати та частоти
        self._create_time_since_and_frequency_features()
        self._check_features()

        # 11. Створення доменно-специфічних ознак
        # Примітка: оригінальна логіка для визначення irregular_spend_col та non_periodic_spend_col
        # може бути не зовсім надійною, якщо порядок або вміст self.features['aggregated_features'] змінюється.
        # Залишено як є згідно з вимогою не змінювати логіку.
        irregular_col_for_domain: str = 'IrregularSpendings'
        if self.features['aggregated_features'] and 'IrregularSpendings' in self.features['aggregated_features']:
            # Оригінальний код: self.features['aggregated_features'][0]
            # Якщо 'IrregularSpendings' є, то його і використовуємо, незалежно від позиції.
            # Якщо його немає, то використовуємо 'IrregularSpendings' як fallback (що вже є в default).
            # Для безпеки, якщо 'IrregularSpendings' дійсно є в списку, використовуємо його.
             irregular_col_for_domain = 'IrregularSpendings' # Залишаємо як є, оскільки зміна логіки не дозволена

        non_periodic_col_for_domain: str = 'NonPeriodicSpendings'
        if len(self.features['aggregated_features']) > 1 and 'NonPeriodicSpendings' in self.features['aggregated_features']:
            # Оригінальний код: self.features['aggregated_features'][1]
            non_periodic_col_for_domain = 'NonPeriodicSpendings' # Залишаємо як є

        self._create_domain_specific_features(
            irregular_spend_col=self.features['aggregated_features'][0] if len(
                self.features['aggregated_features']) > 0 and 'IrregularSpendings' in self.features[
                                                                               'aggregated_features'] else 'IrregularSpendings',
            non_periodic_spend_col=self.features['aggregated_features'][1] if len(
                self.features['aggregated_features']) > 1 and 'NonPeriodicSpendings' in self.features[
                                                                                  'aggregated_features'] else 'NonPeriodicSpendings'
        )
        self._check_features()

        # Фінальна перевірка NaN і заповнення нулями, якщо щось пропущено
        if self.df.isna().sum().sum() > 0:
            self.logger.warning(
                f"Виявлено {self.df.isna().sum().sum()} NaN значень перед фінальним поверненням. Заповнення нулями...")
            self.df.fillna(0, inplace=True)

        self.logger.info("Процес інженерії ознак завершено.")
        return self.df.copy(), self.features.copy()

def main():
    print("Запуск тестового сценарію для FeatureEngineer...")
    try:
        file_path_data_folder = pathlib.Path(__file__).parents[3].resolve() / 'data'
        if not file_path_data_folder.exists():
            current_path = pathlib.Path('.').resolve()
            if (current_path / 'data').exists():
                file_path_data_folder = current_path / 'data'
            elif (current_path.parent / 'data').exists():
                file_path_data_folder = current_path.parent / 'data'
            else:
                raise FileNotFoundError("Папку 'data' не знайдено.")

        df_pickle_path: pathlib.Path = file_path_data_folder / 'intermediate/data_prepare.pkl'
        categories_json_path: pathlib.Path = file_path_data_folder / 'intermediate/columns_cat.json'
        correlated_features_path: pathlib.Path = file_path_data_folder / 'intermediate/correlated_features.txt'  # Не використовується

        engineered_df_output_path: pathlib.Path = file_path_data_folder / 'intermediate/engineered_df.pkl'
        features_dict_output_path: pathlib.Path = file_path_data_folder / 'intermediate/features_dict.json'

        df = pd.read_pickle(df_pickle_path)

        with open(categories_json_path, 'r') as f:
            column_categories_main: dict[str, list[str]] = json.load(f)

        with open(correlated_features_path, 'r') as f:  # Не використовується
            correlated_features_main: list[str] = [line.strip() for line in f.readlines() if line.strip()]

        target_cols_main: list[str] = column_categories_main.get('regular', [])

        engineer = FeatureEngineer()

        engineered_df, features_dict = engineer.engineer_features(
            input_df=df,
            column_categories=column_categories_main,
            target_cols=target_cols_main,
            including_secondary_features=correlated_features_main
        )

        engineered_df.to_pickle(engineered_df_output_path)
        with open(features_dict_output_path, 'w') as f:
            json.dump(features_dict, f, indent=4)  # Додано indent для кращого читання json

        print(f"Розмір DataFrame після інженерії ознак: {engineered_df.shape}")
        print(f"Сума NaN в фінальному DataFrame: {engineered_df.isna().sum().sum()}")

        print(f"\n--- Словник створених ознак (ключі та кількість) ---")
        for key, val_list in features_dict.items():
            print(f"- {key}: ({len(val_list)} ознак)")

        print(f"\n--- Перевірка деяких очікуваних колонок ---")
        if target_cols_main:  # Перевірка, чи є цільові колонки
            expected_example_cols: list[str] = [
                f"{target_cols_main[0]}_log",
                f"{target_cols_main[0]}_lag_{LAGS[0] if LAGS else 'N/A'}",
                f"{target_cols_main[0]}_roll_mean_{WINDOWS[0] if WINDOWS else 'N/A'}",
                'day_of_week_sin',
                'IrregularSpendings'  # Ця колонка створюється, навіть якщо irregular_cols порожні (зі значенням 0.0)
            ]
            for ec in expected_example_cols:
                if 'N/A' in ec:  # Пропускаємо, якщо LAGS або WINDOWS порожні
                    print(f"Колонка '{ec}' не може бути перевірена (LAGS/WINDOWS порожні).")
                    continue
                status: str = "Присутня" if ec in engineered_df.columns else "ВІДСУТНЯ"
                print(f"Колонка '{ec}': {status}")
        else:
            print("Немає цільових колонок для перевірки прикладів.")

    except FileNotFoundError as e_fnf:
        print(f"Помилка шляху до файлу: {e_fnf}. Переконайтеся, що файли даних знаходяться у правильній директорії.")
        print(f"Очікувана структура: <project_root>/data/intermediate/...")
        print(f"Поточна робоча директорія: {pathlib.Path.cwd()}")
    except Exception as e:
        print(f"Під час тестового запуску інженерії ознак сталася помилка: {e}")
        import traceback
        print(traceback.format_exc())

    print("Тестовий сценарій FeatureEngineer завершено.")

if __name__ == '__main__':
    main()