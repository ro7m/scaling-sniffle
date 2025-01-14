import 'package:flutter/foundation.dart' show kDebugMode;
import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'screens/camera_screen.dart';
import 'package:path_provider/path_provider.dart';
import 'package:flutter/services.dart';
import 'dart:io';
import 'dart:convert';

Future<void> main() async {
  try {
    WidgetsFlutterBinding.ensureInitialized();
    
    // Check if in debug mode (simulator/development)
    if (kDebugMode) {
      try {
        final cameras = await availableCameras();
        runApp(MyApp(cameras: cameras));
      } catch (e) {
        // If cameras aren't available (like in simulator), use mock camera
        final mockCamera = CameraDescription(
          name: 'Mock Camera',
          lensDirection: CameraLensDirection.back,
          sensorOrientation: 0,
        );
        runApp(MyApp(cameras: [mockCamera]));
      }
      return;
    }

    // Production mode - normal camera initialization
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

class MyApp extends StatelessWidget {
  final List<CameraDescription> cameras;
  
  const MyApp({Key? key, required this.cameras}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'WUAI OCR App',
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