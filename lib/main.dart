import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'screens/camera_screen.dart';

Future<void> main() async {
  try {
    WidgetsFlutterBinding.ensureInitialized();
    await copyModelsToDocuments();
    final cameras = await availableCameras();
    if (cameras.isEmpty) {
      print('No cameras found');
      runApp(const MyAppError(error: 'No cameras available'));
    } else {
      runApp(MyApp(cameras: cameras));
    }
  } catch (e) {
    print('Error initializing app: $e');
    runApp(MyAppError(error: e.toString()));
  }
}

Future<void> copyModelsToDocuments() async {
  final appDir = await getApplicationDocumentsDirectory();
  final modelsDir = Directory('${appDir.path}/assets/models');
  await modelsDir.create(recursive: true);

  // Copy models from assets to documents
  final manifestContent = await rootBundle.loadString('AssetManifest.json');
  final Map<String, dynamic> manifest = json.decode(manifestContent);
  
  for (String path in manifest.keys) {
    if (path.startsWith('assets/models/')) {
      final filename = path.split('/').last;
      final bytes = await rootBundle.load(path);
      final buffer = bytes.buffer;
      await File('${modelsDir.path}/$filename')
          .writeAsBytes(buffer.asUint8List(bytes.offsetInBytes, bytes.lengthInBytes));
    }
  }
}

class MyApp extends StatelessWidget {
  final List<CameraDescription> cameras;
  
  const MyApp({Key? key, required this.cameras}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Flutter OCR App',
      theme: ThemeData(
        primarySwatch: Colors.blue,
      ),
      home: CameraScreen(cameras: cameras),
    );
  }
}

class MyAppError extends StatelessWidget {
  final String error;

  const MyAppError({Key? key, required this.error}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      home: Scaffold(
        body: Center(
          child: Text('Error: $error'),
        ),
      ),
    );
  }
}