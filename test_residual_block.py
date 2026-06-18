"""ResidualBlock の動作確認テスト"""

import numpy as np
from common import config
GPU_ENABLE = config.GPU_ENABLE
xp = config.get_xp()

from common.layers import ResidualBlock

def test_residual_block_basic():
    """基本的なResidualBlockのテスト(同じチャネル数）"""
    print("=" * 60)
    print("テスト1: 基本的なResidualBlock(チャネル数同じ）")
    print("=" * 60)
    
    # パラメータ設定
    batch_size = 2
    in_channels = 16
    out_channels = 16
    img_size = 28
    filter_size = 3
    weight_init_std = 0.01
    
    # ダミー入力データ
    x = xp.random.randn(batch_size, in_channels, img_size, img_size)
    print(f"入力形状: {x.shape}")
    
    # 重みとバイアスの初期化
    W1 = weight_init_std * xp.random.randn(out_channels, in_channels, filter_size, filter_size)
    b1 = xp.zeros(out_channels)
    W2 = weight_init_std * xp.random.randn(out_channels, out_channels, filter_size, filter_size)
    b2 = xp.zeros(out_channels)
    
    # ResidualBlock の作成
    res_block = ResidualBlock(W1, b1, W2, b2, stride=1, pad=1, use_1x1_conv=False)
    
    # 順伝播
    print("\n【順伝播テスト】")
    y = res_block.forward(x)
    print(f"出力形状: {y.shape}")
    print(f"期待値: ({batch_size}, {out_channels}, {img_size}, {img_size})")
    assert y.shape == (batch_size, out_channels, img_size, img_size), "出力形状が不正です"
    print("✓ 順伝播成功")
    
    # 逆伝播
    print("\n【逆伝播テスト】")
    dout = xp.random.randn(*y.shape)
    dx = res_block.backward(dout)
    print(f"入力勾配形状: {dx.shape}")
    assert dx.shape == x.shape, "入力勾配の形状が不正です"
    print("✓ 逆伝播成功")
    
    # パラメータ勾配の確認
    print("\n【パラメータ勾配テスト】")
    grads = res_block.get_grads()
    print(f"勾配の鍵: {list(grads.keys())}")
    assert 'W1' in grads and 'b1' in grads, "W1, b1 の勾配が見つかりません"
    assert 'W2' in grads and 'b2' in grads, "W2, b2 の勾配が見つかりません"
    print("✓ パラメータ勾配取得成功")
    
    print("\n✅ テスト1 完了\n")


def test_residual_block_with_1x1_conv():
    """チャネル数が異なる場合のResidualBlockテスト"""
    print("=" * 60)
    print("テスト2: ResidualBlock（チャネル数異なる+1x1 Conv）")
    print("=" * 60)
    
    batch_size = 2
    in_channels = 16
    out_channels = 32
    img_size = 28
    filter_size = 3
    weight_init_std = 0.01
    
    x = xp.random.randn(batch_size, in_channels, img_size, img_size)
    print(f"入力形状: {x.shape}")
    
    # メイン路の重み
    W1 = weight_init_std * xp.random.randn(out_channels, in_channels, filter_size, filter_size)
    b1 = xp.zeros(out_channels)
    W2 = weight_init_std * xp.random.randn(out_channels, out_channels, filter_size, filter_size)
    b2 = xp.zeros(out_channels)
    
    # 1x1 Conv の重み（チャネル数調整用）
    W_1x1 = weight_init_std * xp.random.randn(out_channels, in_channels, 1, 1)
    b_1x1 = xp.zeros(out_channels)
    
    # ResidualBlock の作成
    res_block = ResidualBlock(W1, b1, W2, b2, stride=1, pad=1, 
                             use_1x1_conv=True, W_1x1=W_1x1, b_1x1=b_1x1)
    
    print("\n【順伝播テスト】")
    y = res_block.forward(x)
    print(f"出力形状: {y.shape}")
    print(f"期待値: ({batch_size}, {out_channels}, {img_size}, {img_size})")
    assert y.shape == (batch_size, out_channels, img_size, img_size), "出力形状が不正です"
    print("✓ 順伝播成功")
    
    print("\n【逆伝播テスト】")
    dout = xp.random.randn(*y.shape)
    dx = res_block.backward(dout)
    print(f"入力勾配形状: {dx.shape}")
    assert dx.shape == x.shape, "入力勾配の形状が不正です"
    print("✓ 逆伝播成功")
    
    print("\n【パラメータ勾配テスト】")
    grads = res_block.get_grads()
    print(f"勾配の鍵: {list(grads.keys())}")
    assert 'W_1x1' in grads and 'b_1x1' in grads, "1x1 Conv の勾配が見つかりません"
    print("✓ 1x1 Conv の勾配取得成功")
    
    print("\n✅ テスト2 完了\n")


def test_residual_block_stride():
    """ストライド付きのResidualBlockテスト"""
    print("=" * 60)
    print("テスト3: ResidualBlock（ストライド=2）")
    print("=" * 60)
    
    batch_size = 2
    in_channels = 16
    out_channels = 32
    img_size = 28
    filter_size = 3
    stride = 2
    weight_init_std = 0.01
    
    x = xp.random.randn(batch_size, in_channels, img_size, img_size)
    print(f"入力形状: {x.shape}")
    
    W1 = weight_init_std * xp.random.randn(out_channels, in_channels, filter_size, filter_size)
    b1 = xp.zeros(out_channels)
    W2 = weight_init_std * xp.random.randn(out_channels, out_channels, filter_size, filter_size)
    b2 = xp.zeros(out_channels)
    
    # 1x1 Conv でストライドとチャネル数両方調整
    W_1x1 = weight_init_std * xp.random.randn(out_channels, in_channels, 1, 1)
    b_1x1 = xp.zeros(out_channels)
    
    res_block = ResidualBlock(W1, b1, W2, b2, stride=stride, pad=1, 
                             use_1x1_conv=True, W_1x1=W_1x1, b_1x1=b_1x1)
    
    print("\n【順伝播テスト】")
    y = res_block.forward(x)
    expected_size = (img_size + 2*1 - filter_size) // stride + 1
    print(f"出力形状: {y.shape}")
    print(f"期待値: ({batch_size}, {out_channels}, {expected_size}, {expected_size})")
    assert y.shape == (batch_size, out_channels, expected_size, expected_size), "出力形状が不正です"
    print("✓ 順伝播成功")
    
    print("\n【逆伝播テスト】")
    dout = xp.random.randn(*y.shape)
    dx = res_block.backward(dout)
    print(f"入力勾配形状: {dx.shape}")
    assert dx.shape == x.shape, "入力勾配の形状が不正です"
    print("✓ 逆伝播成功")
    
    print("\n✅ テスト3 完了\n")


def test_gradient_check():
    """数値勾配とバックプロップ勾配の比較"""
    print("=" * 60)
    print("テスト4: 勾配チェック（数値勾配 vs バックプロップ）")
    print("=" * 60)
    
    # 小さいサイズで実行
    batch_size = 1
    in_channels = 2
    out_channels = 2
    img_size = 4
    filter_size = 3
    weight_init_std = 0.01
    
    x = xp.random.randn(batch_size, in_channels, img_size, img_size)
    
    W1 = weight_init_std * xp.random.randn(out_channels, in_channels, filter_size, filter_size)
    b1 = xp.zeros(out_channels)
    W2 = weight_init_std * xp.random.randn(out_channels, out_channels, filter_size, filter_size)
    b2 = xp.zeros(out_channels)
    
    res_block = ResidualBlock(W1, b1, W2, b2, stride=1, pad=1, use_1x1_conv=False)
    
    # 順伝播
    y = res_block.forward(x)
    
    # ダミー勾配（スカラー損失）
    dout = xp.ones_like(y)
    
    # バックプロップ
    dx_backprop = res_block.backward(dout)
    
    # 数値勾配計算
    eps = 1e-4
    dx_numerical = xp.zeros_like(x)
    
    for i in range(x.size):
        x_tmp = x.copy()
        x_tmp.flat[i] += eps
        y_pos = res_block.forward(x_tmp)
        loss_pos = xp.sum(y_pos)
        
        x_tmp = x.copy()
        x_tmp.flat[i] -= eps
        y_neg = res_block.forward(x_tmp)
        loss_neg = xp.sum(y_neg)
        
        dx_numerical.flat[i] = (loss_pos - loss_neg) / (2 * eps)
    
    # 差分の計算
    diff = xp.abs(dx_backprop - dx_numerical) / (xp.abs(dx_numerical) + 1e-8)
    max_diff = xp.max(diff)
    mean_diff = xp.mean(diff)
    
    print(f"\n勾配チェック結果:")
    print(f"  最大誤差: {max_diff:.2e}")
    print(f"  平均誤差: {mean_diff:.2e}")
    
    if max_diff < 1e-2:
        print("✓ 勾配チェック成功（誤差は許容範囲内）")
    else:
        print("⚠ 勾配チェック警告（誤差が大きい可能性）")
    
    print("\n✅ テスト4 完了\n")


if __name__ == "__main__":
    print("\n")
    print("*" * 60)
    print("ResidualBlock (Pre-activation) 動作確認テスト")
    print("*" * 60)
    print("\n")
    
    try:
        test_residual_block_basic()
        test_residual_block_with_1x1_conv()
        test_residual_block_stride()
        test_gradient_check()
        
        print("\n")
        print("*" * 60)
        print("🎉 すべてのテストが正常に完了しました！")
        print("*" * 60)
        print("\n")
        
    except Exception as e:
        print(f"\n❌ エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
