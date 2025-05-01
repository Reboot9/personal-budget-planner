from collections import Counter

import numpy as np
from keras.layers import Input, Dense, LSTM, Conv1D, Dense, Flatten, GRU

from keras.models import Model
from keras.src.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.src.layers import Concatenate, BatchNormalization, MaxPooling1D, Bidirectional, MultiHeadAttention, \
    LayerNormalization, Add
from keras.src.optimizers import Adam

from keras.src.optimizers.schedules import ExponentialDecay
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.utils import compute_class_weight
from tensorflow.keras.layers import Dropout, Concatenate, Attention
from tensorflow.keras.optimizers import RMSprop, Nadam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Conv1D, MaxPooling1D
from sklearn.metrics import f1_score
import tensorflow as tf
from sklearn.metrics import f1_score
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam, AdamW
from tensorflow.keras.optimizers.schedules import ExponentialDecay, CosineDecay
from tensorflow.keras.callbacks import EarlyStopping


def focal_loss(gamma=2., alpha=0.25, class_weights=None):
    def focal_loss_fixed(y_true, y_pred):
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
        pt = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
        if class_weights is not None:
            class_weights_tensor = K.constant(class_weights)
            weights = K.sum(class_weights_tensor * y_true, axis=-1)
        else:
            weights = 1.0

        loss = -alpha * K.pow(1 - pt, gamma) * K.log(pt + epsilon) * weights
        return K.sum(loss, axis=-1)

    return focal_loss_fixed


def find_best_threshold(y_true, y_pred_probs, thresholds=np.arange(0.1, 0.9, 0.05)):
    best_thresh = 0.5
    best_f1 = 0

    for t in thresholds:
        y_pred = (y_pred_probs >= t).astype(int)
        f1 = f1_score(y_true, y_pred, average='samples', zero_division=1)  #  'weighted'  'samples'
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t

    return best_thresh, best_f1


def find_best_threshold_per_class(y_true, y_pred_probs, thresholds=np.arange(0.1, 0.9, 0.05)):
    best_thresh = np.zeros(y_pred_probs.shape[1])
    best_f1 = np.zeros(y_pred_probs.shape[1])

    for idx in range(y_pred_probs.shape[1]):
        best_f1_class = 0
        for t in thresholds:
            y_pred = (y_pred_probs[:, idx] >= t).astype(int)
            f1 = f1_score(y_true[:, idx], y_pred, average='binary', zero_division=1)
            if f1 > best_f1_class:
                best_f1_class = f1
                best_thresh[idx] = t

        best_f1[idx] = best_f1_class

    return best_thresh, best_f1


class BudgetPredictionModel:
    def __init__(self, user_id, n_days=30, threshold=0.05,):
        self.user_id = user_id
        self.n_days = n_days
        self.threshold = threshold
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.amounts_scaled = None
        self.categories_encoded = None
        self.encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

    def preprocess_data(self, transactions):
        # Extract the date, amount, category, and temporal features
        data = [{
            'date': transaction.date,
            'amount': max(1, transaction.amount),
            'category': transaction.category.name if transaction.category else 'Other',
            'day_of_week': transaction.date.weekday(),  # 0: Monday, 6: Sunday
            'month': transaction.date.month  # Month of the year
        } for transaction in transactions]

        # Aggregate the amounts by date and category
        aggregated_data = {}
        for entry in data:
            date = entry['date']
            category = entry['category']
            amount = entry['amount']

            if date not in aggregated_data:
                aggregated_data[date] = {}
            if category not in aggregated_data[date]:
                aggregated_data[date][category] = 0
            aggregated_data[date][category] += amount

        # Convert aggregated data back to a list
        aggregated_data_list = []
        for date, categories in aggregated_data.items():
            for category, total_amount in categories.items():
                aggregated_data_list.append({
                    'date': date,
                    'category': category,
                    'amount': total_amount,
                    'day_of_week': date.weekday(),
                    'month': date.month
                })
        # if category isn't present well
        category_counts = Counter([item['category'] for item in aggregated_data_list])
        total = sum(category_counts.values())
        # threshold for rare categories
        category_merging_ratio = 0.015
        rare_categories = {cat for cat, count in category_counts.items() if count / total < category_merging_ratio}

        for item in aggregated_data_list:
            if item['category'] in rare_categories:
                item['category'] = 'Other'

        amounts = [item['amount'] for item in aggregated_data_list]
        categories = [item['category'] for item in aggregated_data_list]
        temporal_features = np.array([[item['day_of_week'] / 6.0, item['month'] / 12.0] for item in aggregated_data_list])

        amounts_scaled = self.scaler.fit_transform(np.array(amounts).reshape(-1, 1))
        categories_encoded = self.encoder.fit_transform(np.array(categories).reshape(-1, 1))

        self.amounts_scaled = amounts_scaled
        self.categories_encoded = categories_encoded
        self.temporal_features = temporal_features

        return amounts_scaled, categories_encoded, aggregated_data_list, temporal_features

    def build_model(self, shape_amount, shape_category, shape_temporal, n_categories):
        input_amount = Input(shape=shape_amount)
        input_category = Input(shape=shape_category)
        input_temporal = Input(shape=shape_temporal)

        x1 = LSTM(128, return_sequences=True, kernel_regularizer=l2(0.001))(input_amount)
        x1 = BatchNormalization()(x1)
        attention_amt = Attention()([x1, x1])

        x2 = GRU(128, return_sequences=True, kernel_regularizer=l2(0.001))(input_category)
        x2 = LayerNormalization()(x2)

        group_attention = MultiHeadAttention(num_heads=4, key_dim=32)(x2, x2)
        x2 = Add()([x2, group_attention])  # Residual connection
        x2 = LayerNormalization()(x2)

        # Second level of attention (subcategory-level)
        subcategory_attention = MultiHeadAttention(num_heads=4, key_dim=32)(x2, x2)
        x2 = Add()([x2, subcategory_attention])  # Residual connection
        x2 = LayerNormalization()(x2)

        attention_cat = x2

        x3 = LSTM(128, return_sequences=True, kernel_regularizer=l2(0.001))(input_temporal)
        x3 = BatchNormalization()(x3)
        attention_tmp = Attention()([x3, x3])

        merged = Concatenate()([attention_amt, attention_cat, attention_tmp])
        x = GRU(128)(merged)
        x = Dropout(0.4)(x)
        x = Dense(128, activation='relu', kernel_regularizer=l2(0.001))(x)

        out_amount = Dense(n_categories, activation='softplus', name='amount')(x)
        out_category = Dense(n_categories, activation='sigmoid', name='category')(x)
        out_temporal = Dense(2, name='temporal')(x)

        model = Model(inputs=[input_amount, input_category, input_temporal],
                      outputs=[out_amount, out_category, out_temporal])
        initial_learning_rate = 0.0001
        lr_schedule = ExponentialDecay(
            initial_learning_rate=initial_learning_rate,
            decay_steps=100000,
            decay_rate=0.96,
            staircase=True
        )
        optimizer = AdamW(learning_rate=lr_schedule, weight_decay=0.01)
        model.compile(optimizer=optimizer,
                      loss={'amount': 'mse', 'category': focal_loss(), 'temporal': 'mse'},
                      metrics={'category': 'accuracy'},
                      loss_weights={'category': 1.8, 'amount': 0.8, 'temporal': 0.2})

        return model

    def train(self, transactions):
        amounts_scaled, categories_encoded, aggregated_data_list, temporal_features = self.preprocess_data(transactions)
        split_index = int(0.8 * len(amounts_scaled))

        X_amt_train, X_amt_val = amounts_scaled[:split_index], amounts_scaled[split_index:]
        X_cat_train, X_cat_val = categories_encoded[:split_index], categories_encoded[split_index:]
        X_tmp_train, X_tmp_val = temporal_features[:split_index], temporal_features[split_index:]

        # Prepare sequences for training data
        X_train_seq_amt, Y_train_amt = [], []
        X_train_seq_cat, Y_train_cat = [], []
        X_train_seq_tmp, Y_train_tmp = [], []

        for i in range(len(X_amt_train) - self.n_days):
            X_train_seq_amt.append(X_amt_train[i:i + self.n_days])
            Y_train_amt.append(X_amt_train[i + self.n_days])

            X_train_seq_cat.append(X_cat_train[i:i + self.n_days])
            Y_train_cat.append(X_cat_train[i + self.n_days])

            X_train_seq_tmp.append(X_tmp_train[i:i + self.n_days])
            Y_train_tmp.append(X_tmp_train[i + self.n_days])

        X_train_amt = np.array(X_train_seq_amt)
        X_train_cat = np.array(X_train_seq_cat)
        X_train_tmp = np.array(X_train_seq_tmp)
        Y_train_amt = np.array(Y_train_amt)
        Y_train_cat = np.array(Y_train_cat)
        Y_train_tmp = np.array(Y_train_tmp)

        # Prepare sequences for validation data
        X_val_seq_amt, Y_val_amt = [], []
        X_val_seq_cat, Y_val_cat = [], []
        X_val_seq_tmp, Y_val_tmp = [], []

        for i in range(len(X_amt_val) - self.n_days):
            X_val_seq_amt.append(X_amt_val[i:i + self.n_days])
            Y_val_amt.append(X_amt_val[i + self.n_days])

            X_val_seq_cat.append(X_cat_val[i:i + self.n_days])
            Y_val_cat.append(X_cat_val[i + self.n_days])

            X_val_seq_tmp.append(X_tmp_val[i:i + self.n_days])
            Y_val_tmp.append(X_tmp_val[i + self.n_days])

        X_val_amt = np.array(X_val_seq_amt)
        X_val_cat = np.array(X_val_seq_cat)
        X_val_tmp = np.array(X_val_seq_tmp)
        Y_val_amt = np.array(Y_val_amt)
        Y_val_cat = np.array(Y_val_cat)
        Y_val_tmp = np.array(Y_val_tmp)

        n_categories = len(self.encoder.categories_[0])

        self.model = self.build_model(X_train_amt.shape[1:], X_train_cat.shape[1:], X_train_tmp.shape[1:], n_categories)
        lr_scheduler = ReduceLROnPlateau(monitor='val_category_loss', factor=0.5, patience=2, min_lr=1e-7)

        early_stopping = EarlyStopping(
            monitor='val_category_loss',
            patience=3,
            restore_best_weights=True,
            verbose=1
        )
        self.model.fit(
            [X_train_amt, X_train_cat, X_train_tmp],
            {'amount': Y_train_amt, 'category': Y_train_cat, 'temporal': Y_train_tmp},
            validation_data=([X_val_amt, X_val_cat, X_val_tmp], {'amount': Y_val_amt, 'category': Y_val_cat, 'temporal': Y_val_tmp}),
            epochs=16,
            batch_size=64,
            callbacks=[lr_scheduler, early_stopping],
            verbose=2
        )
        from sklearn.metrics import classification_report

        y_pred_probs = self.model.predict([X_val_amt, X_val_cat, X_val_tmp], verbose=0)[1]
        best_thresholds, best_f1 = find_best_threshold_per_class(Y_val_cat, y_pred_probs)
        print(f"Best Thresholds per class: {best_thresholds}, Best F1 Scores per class: {best_f1}")

        self.thresholds = best_thresholds
        y_pred = (y_pred_probs >= self.thresholds).astype(int)
        print(classification_report(Y_val_cat, y_pred, target_names=self.encoder.categories_[0], zero_division=0))

    def predict(self, n_days=30):
        if self.amounts_scaled is None or self.categories_encoded is None:
            raise ValueError("Data has not been preprocessed. Please call train first.")

        all_predictions = []

        current_amounts = self.amounts_scaled[-self.n_days:].copy()
        current_categories = self.categories_encoded[-self.n_days:].copy()
        current_temporal = self.temporal_features[-self.n_days:].copy()

        for _ in range(n_days):
            input_amt = np.expand_dims(current_amounts, axis=0)
            input_cat = np.expand_dims(current_categories, axis=0)
            input_tmp = np.expand_dims(current_temporal, axis=0)

            amount_pred, category_pred, temporal_pred = self.model.predict([input_amt, input_cat, input_tmp], verbose=0)
            predicted_amounts = self.scaler.inverse_transform(amount_pred)

            daily_prediction = {}
            for idx, cat_prob in enumerate(category_pred[0]):
                if cat_prob > self.thresholds[idx]:  # Apply per-class threshold
                    category = self.encoder.categories_[0][idx]
                    amount = predicted_amounts[0][idx]
                    daily_prediction[category] = amount

            all_predictions.append(daily_prediction)

            # Update rolling windows
            avg_amount = np.mean(amount_pred, axis=1, keepdims=True)
            current_amounts = np.vstack([current_amounts[1:], avg_amount])

            current_categories = np.vstack([current_categories[1:], category_pred])
            current_temporal = np.vstack([current_temporal[1:], temporal_pred])

        return all_predictions

