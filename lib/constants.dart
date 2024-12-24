class OCRConstants {
  static const List<double> REC_MEAN = [0.694, 0.695, 0.693];
  static const List<double> REC_STD = [0.299, 0.296, 0.301];
  static const List<double> DET_MEAN = [0.798, 0.785, 0.772];
  static const List<double> DET_STD = [0.264, 0.2749, 0.287];
  static const String VOCAB = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!\"#\$%&'()*+,-./:;<=>?@[\\]^_`{|}~°£€¥¢฿àâéèêëîïôùûüçÀÂÉÈÊËÎÏÔÙÛÜÇ";
  static const List<int> TARGET_SIZE = [1024, 1024];
  static const List<int> RECOGNITION_TARGET_SIZE = [32, 128];
}