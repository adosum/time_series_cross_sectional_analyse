is_training = True
pred_len = 10
seq_len = 180
label_len = 120
output_attention = False
enc_in = 7
dec_in = 7
e_layers = 2
d_layers = 2
d_model = 512
n_heads = 8
d_ff = 2048
activation = 'gelu'
c_out = 7
p_hidden_dims = [128, 128]  # hidden layer dimensions of projector (List)
p_hidden_layers = 2  # hidden layer number of projector (int)
embed = 'fixed'  # 'time features encoding, options:[timeF, fixed, learned]'

freq = 'd'  # freq for time features encoding, options:[s:secondly,
# t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly

dropout = 0.1
factor = 5  # attn factor
patience = 3

use_amp = False  # use automatic mixed precision training
train_epochs = 20
model = 'ns_transformer'
model_id = 'daily_test'
features = 'MS'  # M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate
learning_rate = 0.0001
batch_size = 4
timeenc = 0

checkpoints = 'checkpoint/'
