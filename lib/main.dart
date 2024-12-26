import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'screens/camera_screen.dart';
import 'package:path_provider/path_provider.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'dart:convert';
import 'dart:io';

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
  try {
    final appDir = await getApplicationDocumentsDirectory();
    final modelsDir = Directory('${appDir.path}/assets/models');
    
    // Create the directory if it doesn't exist
    if (!await modelsDir.exists()) {
      await modelsDir.create(recursive: true);
    }

    // List of model files to copy
    final modelFiles = [
      'assets/models/rep_fast_base.onnx',
      'assets/models/crnn_mobilenet_v3_large.onnx',
    ];

    // Copy each model file
    for (String assetPath in modelFiles) {
      final filename = assetPath.split('/').last;
      final targetFile = File('${modelsDir.path}/$filename');
      
      // Only copy if the file doesn't exist
      if (!await targetFile.exists()) {
        final byteData = await rootBundle.load(assetPath);
        final buffer = byteData.buffer;
        await targetFile.writeAsBytes(
          buffer.asUint8List(
            byteData.offsetInBytes,
            byteData.lengthInBytes,
          )
        );
        print('Copied $filename to documents directory');
      }
    }
  } catch (e) {
    print('Error copying models: $e');
    throw Exception('Failed to copy models: $e');
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