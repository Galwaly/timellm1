import argparse, os, torch, numpy as np
from data_provider.data_factory import data_provider
from models import TimeLLM

def predict_once(args):
    _, test_loader = data_provider(args, 'test')
    model = TimeLLM.Model(args).float()
    model.load_state_dict(torch.load(args.checkpoint, map_location='cpu'))
    model.eval()

    preds, trues = [], []
    with torch.no_grad():
        for batch_x, batch_y, batch_x_mark, batch_y_mark in test_loader:
            out = model(batch_x, batch_x_mark, None, batch_y_mark)
            f_dim = -1 if args.features == 'MS' else 0
            out   = out[:, -args.pred_len:, f_dim:]
            batch_y = batch_y[:, -args.pred_len:, f_dim:]
            preds.append(out.numpy())
            trues.append(batch_y.numpy())

    preds = np.concatenate(preds, 0)
    trues = np.concatenate(trues, 0)
    _, test_data = data_provider(args, 'test')
    preds = test_data.inverse_transform(preds)
    trues = test_data.inverse_transform(trues)

    os.makedirs(args.result_dir, exist_ok=True)
    np.save(os.path.join(args.result_dir, 'pred.npy'), preds)
    np.save(os.path.join(args.result_dir, 'true.npy'), trues)
    mse = ((preds - trues) ** 2).mean()
    mae = np.abs(preds - trues).mean()
    print(f'Test MSE:{mse:.4f}  MAE:{mae:.4f}')
    with open(os.path.join(args.result_dir, 'metrics.txt'), 'w') as f:
        f.write(f'MSE:{mse}\nMAE:{mae}\n')

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--result_dir',   required=True)

    # 数据 & 任务
    p.add_argument('--data',         default='ETTh1')
    p.add_argument('--root_path',    default='./dataset/ETT-small/')
    p.add_argument('--data_path',    default='ETTh1.csv')
    p.add_argument('--features',     default='M')
    p.add_argument('--seq_len',      type=int, default=512)
    p.add_argument('--label_len',    type=int, default=48)
    p.add_argument('--pred_len',     type=int, default=96)
    p.add_argument('--enc_in',       type=int, default=7)
    p.add_argument('--c_out',        type=int, default=7)

    # 模型
    p.add_argument('--d_model',      type=int, default=32)
    p.add_argument('--d_ff',         type=int, default=128)
    p.add_argument('--llm_layers',   type=int, default=6)
    p.add_argument('--llm_model',    default='LLAMA')
    p.add_argument('--llm_dim',      type=int, default=4096)
    p.add_argument('--patch_len',    type=int, default=16)
    p.add_argument('--stride',       type=int, default=8)
    p.add_argument('--dropout',      type=float, default=0.1)
    p.add_argument('--factor',       type=int, default=3)
    p.add_argument('--prompt_domain',type=int, default=0)

    # 下面 4 行是 data_factory 需要的
    p.add_argument('--embed',        default='timeF')
    p.add_argument('--output_attention', action='store_true')
    p.add_argument('--moving_avg',   type=int, default=25)
    p.add_argument('--seasonal_patterns', default='Monthly')

    args = p.parse_args()
    predict_once(args)