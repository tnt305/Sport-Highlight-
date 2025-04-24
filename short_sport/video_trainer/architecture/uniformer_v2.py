import torch
from models.uniformer_v2.base_model import VisionTransformer
from models.uniformer_v2.initializer import load_state_dict

def uniformerv2_b16(
    pretrained=True, use_checkpoint=False, checkpoint_num=[0],
    t_size=16, dw_reduction=1.5, backbone_drop_path_rate=0., 
    temporal_downsample=True,
    no_lmhra=False, double_lmhra=True,
    return_list=[8, 9, 10, 11], 
    n_layers=4, n_dim=768, n_head=12, mlp_factor=4.0, drop_path_rate=0.,
    mlp_dropout=[0.5, 0.5, 0.5, 0.5], 
    cls_dropout=0.5, num_classes=400,
):
    model = VisionTransformer(
        input_resolution=224,
        patch_size=16,
        width=768,
        layers=12,
        heads=12,
        output_dim=512,
        use_checkpoint=use_checkpoint,
        checkpoint_num=checkpoint_num,
        t_size=t_size,
        dw_reduction=dw_reduction, 
        backbone_drop_path_rate=backbone_drop_path_rate, 
        temporal_downsample=temporal_downsample,
        no_lmhra=no_lmhra,
        double_lmhra=double_lmhra,
        return_list=return_list, 
        n_layers=n_layers, 
        n_dim=n_dim, 
        n_head=n_head, 
        mlp_factor=mlp_factor, 
        drop_path_rate=drop_path_rate, 
        mlp_dropout=mlp_dropout, 
        cls_dropout=cls_dropout, 
        num_classes=num_classes,
    )

    if pretrained:
        print('load pretrained weights')
        state_dict = torch.load(_MODELS["ViT-B/16"], map_location='cpu')
        load_state_dict(model, state_dict)
    return model.eval()


def uniformerv2_l14(
    pretrained=True, use_checkpoint=False, checkpoint_num=[0],
    t_size=16, dw_reduction=1.5, backbone_drop_path_rate=0., 
    temporal_downsample=True,
    no_lmhra=False, double_lmhra=True,
    return_list=[20, 21, 22, 23],
    n_layers=4, n_dim=1024, n_head=16, mlp_factor=4.0, drop_path_rate=0.,
    mlp_dropout=[0.5, 0.5, 0.5, 0.5], 
    cls_dropout=0.5, num_classes=400,
):
    model = VisionTransformer(
        input_resolution=224,
        patch_size=14,
        width=1024,
        layers=24,
        heads=16,
        output_dim=768,
        use_checkpoint=use_checkpoint,
        checkpoint_num=checkpoint_num,
        t_size=t_size,
        dw_reduction=dw_reduction, 
        backbone_drop_path_rate=backbone_drop_path_rate, 
        temporal_downsample=temporal_downsample,
        no_lmhra=no_lmhra,
        double_lmhra=double_lmhra,
        return_list=return_list, 
        n_layers=n_layers, 
        n_dim=n_dim, 
        n_head=n_head, 
        mlp_factor=mlp_factor, 
        drop_path_rate=drop_path_rate, 
        mlp_dropout=mlp_dropout, 
        cls_dropout=cls_dropout, 
        num_classes=num_classes,
    )

    if pretrained:
        print('load pretrained weights')
        state_dict = torch.load(_MODELS["ViT-L/14"], map_location='cpu')
        load_state_dict(model, state_dict)
    return model.eval()


def uniformerv2_l14_336(
    pretrained=True, use_checkpoint=False, checkpoint_num=[0],
    t_size=16, dw_reduction=1.5, backbone_drop_path_rate=0., 
    no_temporal_downsample=True,
    no_lmhra=False, double_lmhra=True,
    return_list=[20, 21, 22, 23],
    n_layers=4, n_dim=1024, n_head=16, mlp_factor=4.0, drop_path_rate=0.,
    mlp_dropout=[0.5, 0.5, 0.5, 0.5], 
    cls_dropout=0.5, num_classes=400,
):
    model = VisionTransformer(
        input_resolution=336,
        patch_size=14,
        width=1024,
        layers=24,
        heads=16,
        output_dim=768,
        use_checkpoint=use_checkpoint,
        checkpoint_num=checkpoint_num,
        t_size=t_size,
        dw_reduction=dw_reduction, 
        backbone_drop_path_rate=backbone_drop_path_rate, 
        no_temporal_downsample=no_temporal_downsample,
        no_lmhra=no_lmhra,
        double_lmhra=double_lmhra,
        return_list=return_list, 
        n_layers=n_layers, 
        n_dim=n_dim, 
        n_head=n_head, 
        mlp_factor=mlp_factor, 
        drop_path_rate=drop_path_rate, 
        mlp_dropout=mlp_dropout, 
        cls_dropout=cls_dropout, 
        num_classes=num_classes,
    )

    if pretrained:
        print('load pretrained weights')
        state_dict = torch.load(_MODELS["ViT-L/14_336"], map_location='cpu')
        load_state_dict(model, state_dict)
    return model.eval()
