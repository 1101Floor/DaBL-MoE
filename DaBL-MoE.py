# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 21:05:07 2026

@author: HP
"""

import tensorflow as tf
import numpy as np
import argparse
from tensorflow.keras.metrics import Metric
import matplotlib.pyplot as plt
import pandas as pd

# -------------------------- 自定义R2评价指标 --------------------------
class R2Score(tf.keras.metrics.Metric):
    def __init__(self, name='r2_score', **kwargs):
        super(R2Score, self).__init__(name=name, **kwargs)
        self.total = self.add_weight(name='total', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        # 计算每个 batch 的 R²
        ss_res = tf.reduce_sum(tf.square(y_true - y_pred))
        ss_tot = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true)))
        r2 = 1.0 - ss_res / (ss_tot + tf.keras.backend.epsilon())
        
        # 累加（注意：这里我们按 batch 累加分子分母更准确，但简单实现可直接平均）
        # 更严谨的做法是累积 ss_res 和 ss_tot，但为简化，我们用加权平均
        batch_size = tf.cast(tf.shape(y_true)[0], dtype=tf.float32)
        self.total.assign_add(r2 * batch_size)
        self.count.assign_add(batch_size)

    def result(self):
        return self.total / self.count

    def reset_state(self):
        self.total.assign(0.0)
        self.count.assign(0.0)

        
class DenormRMSE(tf.keras.metrics.Metric):
    def __init__(self, mean, std, name="denorm_rmse", **kwargs):
        super(DenormRMSE, self).__init__(name=name, **kwargs)
        self.mean = tf.constant(float(mean), dtype=tf.float32)   # Y 列均值
        self.std = tf.constant(float(std), dtype=tf.float32)     # Y 列标准差
        self.sum_sq = self.add_weight(name="sum_sq", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
        y_pred = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)
        # 反标准化：还原到原始量纲 y_raw = y_norm * std + mean
        y_true_raw = y_true * self.std + self.mean
        y_pred_raw = y_pred * self.std + self.mean
        self.sum_sq.assign_add(tf.reduce_sum(tf.square(y_true_raw - y_pred_raw)))
        self.count.assign_add(tf.cast(tf.shape(y_true)[0], tf.float32))
    def result(self):
        return tf.sqrt(self.sum_sq / (self.count + tf.keras.backend.epsilon()))
    def reset_state(self):
        for s in self.variables:
            s.assign(tf.zeros_like(s))

class DenormMAE(tf.keras.metrics.Metric):
    def __init__(self, mean, std, name="denorm_mae", **kwargs):
        super(DenormMAE, self).__init__(name=name, **kwargs)
        self.mean = tf.constant(float(mean), dtype=tf.float32)   # Y 列均值
        self.std = tf.constant(float(std), dtype=tf.float32)     # Y 列标准差
        self.sum_abs = self.add_weight(name="sum_abs", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
        y_pred = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)
        # 反标准化：还原到原始量纲
        y_true_raw = y_true * self.std + self.mean
        y_pred_raw = y_pred * self.std + self.mean
        self.sum_abs.assign_add(tf.reduce_sum(tf.abs(y_true_raw - y_pred_raw)))
        self.count.assign_add(tf.cast(tf.shape(y_true)[0], tf.float32))
    def result(self):
        return self.sum_abs / (self.count + tf.keras.backend.epsilon())
    def reset_state(self):
        for s in self.variables:
            s.assign(tf.zeros_like(s))
        
        
# -------------------------- 全部子模块定义 --------------------------
class TemporalFeatureMapping(tf.keras.models.Model):
    def __init__(self, feature_num, seq_len, num_map, map_num=4, D=30, activation='sig'):
        super(TemporalFeatureMapping, self).__init__()
        self.feature_num = feature_num
        self.num_map = num_map
        self.map_num = map_num
        self.D = D
        self.activation = activation
        self.initializer_w = tf.keras.initializers.GlorotUniform(seed=42)
        self.initializer_b = tf.keras.initializers.Zeros()

        self.W_m = []
        self.b_m = []
        self.W_r = []
        for i in range(map_num):
            wm = self.add_weight(shape=(feature_num, num_map), initializer=self.initializer_w, trainable=True, name=f"Wm_{i}")
            bm = self.add_weight(shape=(num_map,), initializer=self.initializer_b, trainable=True, name=f"bm_{i}")
            wr = self.add_weight(shape=(args.seq_len * num_map, D), initializer=self.initializer_w, trainable=True, name=f"Wr_{i}")
            self.W_m.append(wm)
            self.b_m.append(bm)
            self.W_r.append(wr)

        self.pos_encoding = self._build_sinusoidal_pos_enc(seq_len, num_map)

    def _build_sinusoidal_pos_enc(self, seq_len, d_model):
        pos = np.arange(seq_len)[:, np.newaxis]
        i = np.arange(d_model)[np.newaxis, :]
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        angle_rads = pos * angle_rates
        pe = np.zeros((seq_len, d_model))
        pe[:, 0::2] = np.sin(angle_rads[:, 0::2])
        pe[:, 1::2] = np.cos(angle_rads[:, 1::2])
        pe = pe[np.newaxis, ...]
        return tf.constant(pe, dtype=tf.float32)

    def _activation(self, X):
        if self.activation == 'sig':
            return tf.sigmoid(X)
        elif self.activation == 'tanh':
            return tf.tanh(X)
        else:
            raise ValueError("Unsupported activation function")

    def call(self, X):
        T = args.seq_len
        H = []
        for i in range(self.map_num):
            zi = self._activation(tf.matmul(X, self.W_m[i]) + self.b_m[i])
            P_T = self.pos_encoding[:, :T, :]
            Z_add_pos = zi + P_T
            # vec_z = tf.reshape(Z_add_pos, [batch_size, -1])
            vec_z = tf.reshape(Z_add_pos, [-1, Z_add_pos.shape[1] * Z_add_pos.shape[2]])
            # print(vec_z.shape, "vec_z", Z_add_pos.shape[1], Z_add_pos.shape[2])
            vec_z = tf.matmul(vec_z, self.W_r[i])
            H.append(vec_z)
        Z_n = tf.stack(H, axis=1)
        return Z_n

class FeatureEnhanceLayer(tf.keras.models.Model):
    def __init__(self, m_enhance: int, D_dim: int, activation="tanh"):
        super().__init__()
        self.m_enhance = m_enhance
        self.D_dim = D_dim
        self.activation = tf.keras.activations.get(activation)

    def build(self, input_shape):
        self.n_groups = int(input_shape[-2])
        self.in_dim = self.n_groups * self.D_dim
        self.W_h_list = []
        self.b_h_list = []
        for j in range(self.m_enhance):
            w_h = self.add_weight(shape=(self.in_dim, self.D_dim), initializer="glorot_uniform", trainable=True, name=f"W_h_{j}")
            b_h = self.add_weight(shape=(self.D_dim,), initializer="zeros", trainable=True, name=f"b_h_{j}")
            self.W_h_list.append(w_h)
            self.b_h_list.append(b_h)
        super().build(input_shape)

    def call(self, Z_n):
        # M = tf.shape(Z_n)[0]
        # z = tf.reshape(Z_n, [M, -1])
        z = tf.reshape(Z_n, [-1, Z_n.shape[1] * Z_n.shape[2]])
        H_j_list = []
        for j in range(self.m_enhance):
            linear = tf.matmul(z, self.W_h_list[j]) + self.b_h_list[j]
            H_j = self.activation(linear)
            H_j_list.append(H_j)
        H_m = tf.stack(H_j_list, axis=1)
        return H_m

class CrossAttentionExpertEnhance(tf.keras.models.Model):
    def __init__(self, d_k: int, D_dim: int):
        super().__init__()
        self.d_k = d_k
        self.sqrt_dk = tf.math.sqrt(tf.cast(d_k, tf.float32))
        self.proj_q = tf.keras.layers.Dense(d_k, use_bias=False, name="W_Qc")
        self.proj_k = tf.keras.layers.Dense(d_k, use_bias=False, name="W_Kc")
        self.proj_v = tf.keras.layers.Dense(d_k, use_bias=False, name="W_Vc")
        self.proj_out = tf.keras.layers.Dense(units=D_dim, name="W_Oc")

    def call(self, H_m, Z_n):
        Q_c = self.proj_q(H_m)
        K_c = self.proj_k(Z_n)
        V_c = self.proj_v(Z_n)
        attn_score = tf.matmul(Q_c, tf.transpose(K_c, perm=[0, 2, 1])) / self.sqrt_dk
        alpha_c = tf.nn.softmax(attn_score, axis=-1)
        U_tilde_m = tf.matmul(alpha_c, V_c)
        U_m = self.proj_out(U_tilde_m)
        return U_m

class ExpertGateLayer(tf.keras.models.Model):
    def __init__(self, m_enhance: int):
        super().__init__()
        self.m_enhance = m_enhance
        self.gate_dense = tf.keras.layers.Dense(m_enhance, use_bias=True, name="gate_linear")

    def call(self, Z_n):
        C = tf.reduce_mean(Z_n, axis=1)
        logits = self.gate_dense(C)
        G_m = tf.nn.softmax(logits, axis=-1)
        return G_m

#class ExpertGateLayer(tf.keras.models.Model):
#    def __init__(self, m_enhance: int):
#        super().__init__()
#        self.m_enhance = m_enhance
#        # 删除原来可训练gate_dense层，不再学习门控
#        # self.gate_dense = tf.keras.layers.Dense(m_enhance, use_bias=True, name="gate_linear")

#    def call(self, Z_n):
#        c
#        # 均匀权重：每个专家权重 1/m_enhance
#        uniform_weight = tf.ones((batch_size, self.m_enhance), dtype=tf.float32) / self.m_enhance
#        G_m = uniform_weight
#        return G_m

#class ExpertGateLayer(tf.keras.models.Model):
#    def __init__(self, m_enhance: int):
#        super().__init__()
#        self.m_enhance = m_enhance

#    def call(self, Z_n):
        # 原始输出shape: [batch, m_enhance]，和学习门控输出shape严格对齐
#        ones_vec = tf.ones((self.m_enhance,), dtype=Z_n.dtype) / self.m_enhance
        # 从Z_n拿第0维动态batch，构造目标shape，不单独赋值batch_size变量
#        G_m = tf.broadcast_to(ones_vec, shape=tf.concat([tf.shape(Z_n)[:1], [self.m_enhance]], axis=0))
#        return G_m


class GateExpertFusionLayer(tf.keras.models.Model):
    def __init__(self):
        super().__init__()

    def call(self,):
        #print(G_m.shape, "++++++++++++++++++G_m shape++++++++++++++++++")
        G_expand = tf.expand_dims()
        #print(G_expand.shape, "++++++++++++++++++G_m shape++++++++++++++++++")
        E_m = U_m * G_expand
        return E_m

class FeatureNodeSelfAttention(tf.keras.models.Model):
    def __init__(self, d_k: int, D: int):
        super().__init__()
        self.d_k = d_k
        self.sqrt_dk = tf.math.sqrt(tf.cast(d_k, tf.float32))
        self.proj_qs = tf.keras.layers.Dense(d_k, use_bias=False, name="W_Qs")
        self.proj_ks = tf.keras.layers.Dense(d_k, use_bias=False, name="W_Ks")
        self.proj_vs = tf.keras.layers.Dense(d_k, use_bias=False, name="W_Vs")
        self.proj_os = tf.keras.layers.Dense(D, use_bias=False, name="W_Os")

    def call(self):
        Q_s = self.proj_qs()
        K_s = self.proj_ks()
        V_s = self.proj_vs()
        attn_score = tf.matmul(Q_s, tf.transpose(K_s, perm=[0, 2, 1])) / self.sqrt_dk
        alpha_s = tf.nn.softmax(attn_score, axis=-1)
        attn_out = tf.matmul(alpha_s, V_s)
        F_n = self.proj_os(attn_out)
        return F_n

class FeatureConcatLayer(tf.keras.models.Model):
    def __init__(self, out_dim):
        super().__init__()
        self.proj_out = tf.keras.layers.Dense(units=out_dim)

    def call(self, F_n, E_m):
        vec_F = tf.reshape(F_n, [-1, F_n.shape[1] * F_n.shape[2]])
        # (3.42) vec(E^m) 展平 m、D 维度 -> (M, m*D)
        vec_E = tf.reshape(E_m, [-1, E_m.shape[1] * E_m.shape[2]])
        
        A = tf.concat([vec_F, vec_E], axis=-1)
        
        pred_out = self.proj_out(A)
        return pred_out

# -------------------------- 整体集成主干层 --------------------------
class MultiModelIntegration(tf.keras.layers.Layer):
    def __init__(self, args):
        super().__init__()
        # 时序映射层参数
        self.num_map = args.num_dim
        self.map_num = args.map_num
        self.D = args.d_model
        self.act_mapping = args.act_mapping
        self.act_enhance = args.act_enhance
        self.m_enhance = args.m_enhance
        self.d_k = args.d_k
        self.out_num = args.out_num

        # 子模块实例化，全部从args读取
        self.temporal_map = TemporalFeatureMapping(
            feature_num=args.feature_num,
            seq_len=args.seq_len,
            num_map=self.num_map,
            map_num=self.map_num,
            D=self.D,
            activation=self.act_mapping
        )
        self.self_attn_feat = FeatureNodeSelfAttention(d_k=self.d_k, D=self.D)
        self.feature_enhance = FeatureEnhanceLayer(m_enhance=self.m_enhance, D_dim=self.D, activation=self.act_enhance)
        self.cross_attn = CrossAttentionExpertEnhance(d_k=self.d_k, D_dim=self.D)
        self.gate_net = ExpertGateLayer(m_enhance=self.m_enhance)
        self.fusion = GateExpertFusionLayer()
        self.concat_pred = FeatureConcatLayer(out_dim=self.out_num)

    def call(self, X):
        Z_n = self.temporal_map(X)
        F_n = self.self_attn_feat(Z_n)
        H_m = self.feature_enhance(Z_n)
        U_m = self.cross_attn(H_m, Z_n)
        G_m = self.gate_net(Z_n)
        
        E_m = self.fusion(U_m, G_m)
        y_pred = self.concat_pred(F_n, E_m)
        
        return y_pred


# ===================== 可视化工具函数 =====================
def plot_training_curves(history, save_path="./figures/training_metrics.png"):
    """绘制2行3列训练指标仪表盘：Loss、RMSE、MAE、MAPE、R²"""
    hist = history.history
    epochs = list(range(1, len(hist["loss"]) + 1))
    fig, axes = plt.subplots(2, 2, figsize=(18, 10), dpi=150)
    # fig.suptitle("Model Training Metrics Curve", fontsize=14, weight="bold")
    axes = axes.flatten()

    metrics = [
        ("loss", "val_loss", "MSE Loss"),
        ("root_mean_squared_error", "val_root_mean_squared_error", "RMSE"),
        ("mean_absolute_error", "val_mean_absolute_error", "MAE"),
        # ("mean_absolute_percentage_error", "val_mean_absolute_percentage_error", "MAPE(%)"),
        # ("r2_score", "val_r2_score", "R² Score"),
    ]

    for idx, (train_key, val_key, title) in enumerate(metrics):
        ax = axes[idx]
        ax.plot(epochs, hist[train_key], label="Train", color="#1f77b4", linewidth=1.2)
        ax.plot(epochs, hist[val_key], label="Val", color="#ff7f0e", linewidth=1.2)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Epoch", fontsize=8)
        ax.set_ylabel(title, fontsize=8)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

    # 隐藏多余子图
    axes[-1].axis("off")
    plt.tight_layout()
    # plt.savefig(save_path, bbox_inches="tight")
    plt.show()
    # print(f"训练曲线已保存至: {save_path}")

def plot_pred_true_scatter(model, X_test, y_test, save_path="./figures/pred_true_scatter.png"):
    """验证集真值vs预测值散点图，带y=x基准线"""
    y_pred = model.predict(X_test, verbose=0)
    y_true = y_test.reshape(-1)
    y_pred = y_pred.reshape(-1)

    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    ax.scatter(y_true, y_pred, alpha=0.6, s=8, c="#2ca02c", label="Pred vs True")
    # 基准线 y=x
    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_pred))
    ax.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=1.5, label="y=x Baseline")
    ax.set_xlabel("True Value")
    ax.set_ylabel("Predicted Value")
    ax.set_title("True vs Predicted Scatter Plot")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    # plt.savefig(save_path, bbox_inches="tight")
    plt.show()
    # print(f"预测真值散点图已保存至: {save_path}")

# ===================== 数据 =====================
# data = pd.read_csv("H:/油耗/data/ship_1_x_d_zscore.csv"
#                         , index_col=0, header=0)
# timefun_ = lambda x: datetime.datetime.strptime(x,"%Y-%m-%d %H:%M:%S")
# data['measurement_time'] = data['measurement_time'].apply(timefun_)

data_leg1 = pd.read_csv("data_leg_1.csv"
                        , index_col=0, header=0)
# 删除这三列（axis=1 表示按列删，errors='ignore' 防止列不存在时报错）
#data_leg1 = data_leg1.drop(
#    columns=['shaft_power(kW)', 'shaft_torque(kNm)', 'shaft_revolution(RPM)'],
#    errors='ignore')

fuel_target = ['me_fule_consumption']#, 'boiler_fule_consumption' 'ae_fule_consumption'

# 获取样本
def create_sliding_window_samples(df, window_size, step):
    features = []
    labels = []
    for i in range(0, len(df) - window_size, step):
        feature = df.iloc[i:i+window_size, 1:-1].values # 当取到-1时，表示的是Y不作为变量输入到模型
        label = df.loc[i+window_size, fuel_target]  # Y作为标签
        features.append(feature)
        labels.append(label)
    return np.array(features), np.array(labels)

def multi_col_standardize(data):
    """
    多列分别Z-Score标准化
    :param train_data: 训练集标签/特征，shape [样本数, 列数]
    :param test_data: 测试集标签/特征，shape [样本数, 列数]
    :return: train_std, test_std, mean_arr, std_arr
    """
    # 逐列计算训练集均值、标准差
    mean_arr = np.mean(data, axis=0)
    std_arr = np.std(data, axis=0)
    # 防止除0（列全部数值相同）
    std_arr[std_arr < 1e-8] = 1e-8
    
    train_std = (data - mean_arr) / std_arr
    return train_std, mean_arr, std_arr

def inverse_multi_standardize(data_std, mean_arr, std_arr):
    """逆标准化，还原原始尺度"""
    return data_std * std_arr + mean_arr


parser = argparse.ArgumentParser(description='Tensorflow Integrated MoE-BLS Model')
# 1. 实验数据参数
parser.add_argument('--seq_len', default=12, type=int, help="Input sequence length T")
parser.add_argument('--feature_num', default=13, type=int, help='Input raw feature dimension d')
# parser.add_argument('--pred_len', default=1, type=int, help='Output sequence length')
parser.add_argument('--out_num', default=1, type=int, help='Output prediction dimension')

# 2. 子模型结构超参（全部新增到命令行）
# 时序特征映射层 TemporalFeatureMapping
parser.add_argument('--num_dim', default=128, type=int, help='p: single mapping node dimension')
parser.add_argument('--map_num', default=8, type=int, help='n: number of feature groups')
parser.add_argument('--act_mapping', default="tanh", type=str, help='activation for temporal mapping: sig/tanh')
# 特征增强层 FeatureEnhanceLayer
parser.add_argument('--m_enhance', default=8, type=int, help='m: number of expert groups')
parser.add_argument('--act_enhance', default="tanh", type=str, help='activation for feature enhance layer')
# 注意力通用维度
parser.add_argument('--d_model', default=128, type=int, help='D: unified hidden dimension')
parser.add_argument('--d_k', default=128, type=int, help='Q/K projection dimension for self/cross attention')

# 3. 训练超参
parser.add_argument('--dropout', default=0.1, type=float, help="Dropout rate")
parser.add_argument('--lr', default=0.0001, type=float, help='Learning rate')
parser.add_argument('--epochs', type=int, default=1, help="Total train epochs")
parser.add_argument('--bs', default=128, type=int, help="Training batch size")

args = parser.parse_args(args=[])

# ===================== 命令行参数配置（新增所有子模型超参） =====================
if __name__ == "__main__":
    # for i in [3, 6, 12, 18, 30, 42]:
    for i in [12]:
        window_size = i
        args.seq_len = i
        # -------------------------- 模型构建（完全匹配你指定写法） --------------------------
        enc_input = tf.keras.layers.Input(shape=[args.seq_len, args.feature_num], name='enc_input')
        output = MultiModelIntegration(args)(enc_input)
        model = tf.keras.models.Model(inputs=[enc_input], outputs=[output])
    
        # -------------------------- 模型编译（完全复用你指定指标） --------------------------
        
    
        # window_size = 12
        # args.seq_len = 12
        # 创建滑动窗口样本和标签
        # features, labels = create_sliding_window_samples(data, window_size, step=1)
        
        # del data
        # window_size = 12
        # args.seq_len = 12
        # 创建滑动窗口样本和标签
        features, labels = create_sliding_window_samples(data_leg1, window_size, step=1)
    
        indecies = np.arange(features.shape[0])
        np.random.shuffle(indecies)
        s_i, s_o = features[indecies], labels[indecies].astype(np.float32)
        s_i, y_mean, y_std = multi_col_standardize(s_i)
        s_o1, y_mean_so, y_std_so = multi_col_standardize(s_o)
    
        # del features, labels
        # # 分割数据集为训练集和测试集
        train_ratio = 0.8
        split_index = int(train_ratio * s_i.shape[0])
    
        X_train, X_test = s_i[:split_index], s_i[split_index:]
        y_train, y_test = s_o[:split_index], s_o[split_index:]
        
        del s_i, s_o
        # def generate_dataset(n_samples):
        #     X = np.random.randn(n_samples, args.seq_len, args.feature_num).astype(np.float32)
        #     y = np.random.randn(n_samples, args.out_num).astype(np.float32)
        #     return X, y
    
        # X_train, y_train = generate_dataset(n_samples=2000)
        # X_test, y_test = generate_dataset(n_samples=400)
        
        model.compile(
            loss=tf.keras.losses.MeanSquaredError(),
            optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
            metrics=[
                tf.keras.metrics.RootMeanSquaredError(),
                tf.keras.metrics.MeanAbsoluteError(),
                DenormRMSE(mean=y_mean_so, std=y_std_so), 
                DenormMAE(mean=y_mean_so, std=y_std_so), 
                # tf.keras.metrics.MeanAbsolutePercentageError(),
                R2Score()
            ]
        )
        
        # -------------------------- 训练逻辑与你提供代码完全一致 --------------------------
        history = model.fit(
            X_train, y_train,
            epochs=args.epochs,
            batch_size=args.bs,
            validation_data=(X_test, y_test)
        )
        
        pred = model.predict(features)
        result1 = pd.DataFrame(np.c_[pred, labels])
        #result1.to_csv('result_DAEMBLS_leg1_128_1_gateavg_{filename}.csv'.format(filename=i))
    
        model.summary()
        import time
        start_time = time.time()
        a = model.predict(X_test)
        end_time = time.time()
        
        print("推理时间：")
        print((end_time - start_time))# / X_test.shape[0]
        
        # ===================== 训练完成后自动可视化 =====================
        plot_training_curves(history)
        plot_pred_true_scatter(model, X_test, y_test)
        
        result = pd.DataFrame(history.history)
        result.to_csv('DAEMBLS_STDandORg_{filename}.csv'.format(filename=i))
    
    