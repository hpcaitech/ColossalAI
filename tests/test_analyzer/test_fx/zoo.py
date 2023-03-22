import timm.models as tmm
import torchvision.models as tm

# input shape: (batch_size, 3, 224, 224)
tm_models = [
    tm.alexnet,
    tm.convnext_base,
    tm.densenet121,
    # tm.efficientnet_v2_s,
    # tm.googlenet,   # output bad case
    # tm.inception_v3,  # bad case
    tm.mobilenet_v2,
    tm.mobilenet_v3_small,
    tm.mnasnet0_5,
    tm.resnet18,
    tm.regnet_x_16gf,
    tm.resnext50_32x4d,
    tm.shufflenet_v2_x0_5,
    tm.squeezenet1_0,
    # tm.swin_s,  # fx bad case
    tm.vgg11,
    tm.vit_b_16,
    tm.wide_resnet50_2,
]

tmm_models = [
    tmm.beit_base_patch16_224,
    tmm.beitv2_base_patch16_224,
    tmm.cait_s24_224,
    tmm.coat_lite_mini,
    tmm.convit_base,
    tmm.deit3_base_patch16_224,
    tmm.dm_nfnet_f0,
    tmm.eca_nfnet_l0,
    tmm.efficientformer_l1,
    # tmm.ese_vovnet19b_dw,
    tmm.gmixer_12_224,
    tmm.gmlp_b16_224,
    # tmm.hardcorenas_a,
    tmm.hrnet_w18_small,
    tmm.inception_v3,
    tmm.mixer_b16_224,
    tmm.nf_ecaresnet101,
    tmm.nf_regnet_b0,
    # tmm.pit_b_224,  # pretrained only
    # tmm.regnetv_040,
    # tmm.skresnet18,
    # tmm.swin_base_patch4_window7_224,     # fx bad case
    # tmm.tnt_b_patch16_224,    # bad case
    tmm.vgg11,
    tmm.vit_base_patch16_18x2_224,
    tmm.wide_resnet50_2,
]
