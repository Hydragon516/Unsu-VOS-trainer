DATA = {
    'data_root': "/SSD/minhyeok/dataset/VOS",

    'pretrain': "DUTS_train",

    'best_pretrained_model': "./log/2022-05-19 13:18:14/model/best_model.pth",
    'DAVIS_train_main': "DAVIS_train",
    'DAVIS_train_sub': "YTVOS_train", # or None
    'DAVIS_val': "DAVIS_test",
    
    'best_model': "./log/2022-05-19 13:25:18/model/best_model.pth",
    'FBMS_test': "FBMS_test",
    'YTobj_test': "YTobj_test",
}

TRAIN = {
    'GPU': "0, 1",
    'epoch': 200,
    'learning_rate': 1e-4,
    'print_freq': 50,
    'batch_size': 24,
    'img_size': 352
}