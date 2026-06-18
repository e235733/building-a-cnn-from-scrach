"""NN_Model に ResidualBlock を統合したテスト"""

import numpy as np
from common import config
GPU_ENABLE = config.GPU_ENABLE
xp = config.get_xp()

from nn import NN_Model

def test_nn_model_with_residual_block():
    """ResidualBlock を含むNN_Modelのテスト"""
    print("=" * 60)
    print("NN_Model with ResidualBlock テスト")
    print("=" * 60)
    
    # ResidualBlock を含む layer_config
    layer_config = [
        ['Conv', 16, 5, 0, 1],              # Conv(1->16, 5x5, pad=0, stride=1)
        ['Relu'],
        ['ResidualBlock', 16, 3, 1, 1],     # ResidualBlock(チャネル数同じ)
        ['Pool', 2, 2, 2],                  # Pooling
        ['ResidualBlock', 32, 3, 2, 1],     # ResidualBlock(チャネル数増加 + stride=2)
        ['Affine', 100],                    # 全結合層
        ['Relu'],
        ['Affine', 10]                      # 出力層
    ]
    
    # モデル作成
    print("\n【モデル構築】")
    model = NN_Model(
        input_dim=(1, 28, 28),
        layer_config=layer_config,
        last_layer='Softmax',
        weight_init_type='he',
        use_batchnorm=False
    )
    print(f"レイヤ数: {len(model.layers)}")
    print(f"パラメータ数: {len(model.params)}")
    print(f"\nレイヤー一覧:")
    for layer_name in model.layers.keys():
        print(f"  - {layer_name}")
    
    # テストデータ
    batch_size = 4
    x = xp.random.randn(batch_size, 1, 28, 28)
    t = xp.array([0, 1, 2, 3])
    
    print(f"\n入力形状: {x.shape}")
    print(f"ラベル: {t}")
    
    # 順伝播テスト
    print("\n【順伝播テスト】")
    model.is_training = False
    y = model.predict(x)
    print(f"予測出力形状: {y.shape}")
    assert y.shape == (batch_size, 10), "予測出力形状が不正"
    print("✓ 予測順伝播成功")
    
    # 損失計算
    print("\n【損失計算】")
    model.is_training = True
    loss = model.loss(x, t)
    print(f"損失値: {loss:.6f}")
    assert loss > 0, "損失値が不正"
    print("✓ 損失計算成功")
    
    # 勾配計算テスト
    print("\n【勾配計算テスト】")
    grads = model.gradient(x, t)
    print(f"勾配数: {len(grads)}")
    print(f"勾配の鍵(先頭10個): {list(grads.keys())[:10]}")
    
    # 勾配の存在確認
    assert 'W1' in grads and 'b1' in grads, "最初のパラメータの勾配がない"
    print("✓ 勾配計算成功")
    
    # 勾配の値をチェック
    print("\n【勾配値チェック】")
    non_zero_grads = 0
    for key, grad in grads.items():
        if xp.sum(xp.abs(grad)) > 0:
            non_zero_grads += 1
    print(f"0でない勾配: {non_zero_grads} / {len(grads)}")
    assert non_zero_grads > 0, "勾配が全て0"
    print("✓ 勾配値が適切に計算されている")
    
    # 正解率計算
    print("\n【正解率計算】")
    model.is_training = False
    accuracy = model.accuracy(x, t)
    print(f"正解率: {accuracy:.4f}")
    print("✓ 正解率計算成功")
    
    print("\n✅ テスト完了\n")


def test_nn_model_different_architectures():
    """異なるアーキテクチャのテスト"""
    print("=" * 60)
    print("異なるアーキテクチャのテスト")
    print("=" * 60)
    
    test_cases = [
        {
            'name': 'シンプルCNN',
            'config': [
                ['Conv', 16, 5, 0, 1],
                ['Relu'],
                ['Pool', 2, 2, 2],
                ['Affine', 100],
                ['Relu'],
                ['Affine', 10]
            ]
        },
        {
            'name': 'ResidualBlock 1回',
            'config': [
                ['Conv', 16, 5, 0, 1],
                ['Relu'],
                ['ResidualBlock', 16, 3, 1, 1],
                ['Pool', 2, 2, 2],
                ['Affine', 100],
                ['Relu'],
                ['Affine', 10]
            ]
        },
        {
            'name': 'ResidualBlock 複数回',
            'config': [
                ['Conv', 16, 5, 0, 1],
                ['ResidualBlock', 16, 3, 1, 1],
                ['ResidualBlock', 16, 3, 1, 1],
                ['Pool', 2, 2, 2],
                ['ResidualBlock', 32, 3, 1, 1],
                ['Affine', 100],
                ['Relu'],
                ['Affine', 10]
            ]
        },
        {
            'name': 'ResidualBlock ストライド付き',
            'config': [
                ['Conv', 16, 5, 0, 1],
                ['ResidualBlock', 16, 3, 1, 1],
                ['ResidualBlock', 32, 3, 2, 1],  # stride=2, チャネル数増加
                ['Pool', 2, 2, 2],
                ['Affine', 100],
                ['Relu'],
                ['Affine', 10]
            ]
        }
    ]
    
    for test in test_cases:
        print(f"\n【{test['name']}】")
        try:
            model = NN_Model(
                input_dim=(1, 28, 28),
                layer_config=test['config'],
                weight_init_type='he'
            )
            
            x = xp.random.randn(2, 1, 28, 28)
            t = xp.array([0, 1])
            
            model.is_training = False
            y = model.predict(x)
            
            model.is_training = True
            loss = model.loss(x, t)
            
            grads = model.gradient(x, t)
            
            print(f"  レイヤ数: {len(model.layers)}")
            print(f"  パラメータ数: {len(model.params)}")
            print(f"  勾配数: {len(grads)}")
            print(f"  損失: {loss:.6f}")
            print(f"  ✓ 成功")
            
        except Exception as e:
            print(f"  ❌ エラー: {e}")
    
    print("\n✅ 複数アーキテクチャテスト完了\n")


if __name__ == "__main__":
    print("\n")
    print("*" * 60)
    print("NN_Model ResidualBlock 統合テスト")
    print("*" * 60)
    print("\n")
    
    try:
        test_nn_model_with_residual_block()
        test_nn_model_different_architectures()
        
        print("\n")
        print("*" * 60)
        print("🎉 すべてのテストが正常に完了しました！")
        print("*" * 60)
        print("\n")
        
    except Exception as e:
        print(f"\n❌ エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
