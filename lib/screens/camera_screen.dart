import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:path_provider/path_provider.dart';
import 'package:path/path.dart' show join;
import 'package:permission_handler/permission_handler.dart';
import 'preview_screen.dart';

class CameraScreen extends StatefulWidget {
  final List<CameraDescription> cameras;

  const CameraScreen({Key? key, required this.cameras}) : super(key: key);

  @override
  CameraScreenState createState() => CameraScreenState();
}

class CameraScreenState extends State<CameraScreen> {
  late CameraController _controller;
  late Future<void> _initializeControllerFuture;
  bool _isCameraPermissionGranted = false;

  @override
  void initState() {
    super.initState();
    _requestCameraPermission();
  }

  Future<void> _requestCameraPermission() async {
    final status = await Permission.camera.request();
    setState(() {
      _isCameraPermissionGranted = status == PermissionStatus.granted;
    });
    if (_isCameraPermissionGranted) {
      _initializeCamera();
    }
  }

  Future<void> _initializeCamera() async {
    _controller = CameraController(
      widget.cameras[0],
      ResolutionPreset.medium,
    );
    _initializeControllerFuture = _controller.initialize();
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  Future<void> _takePicture() async {
    try {
      await _initializeControllerFuture;
      final path = join(
        (await getTemporaryDirectory()).path,
        '${DateTime.now()}.png',
      );
      
      final image = await _controller.takePicture();
      
      if (!mounted) return;

      await Navigator.of(context).push(
        MaterialPageRoute(
          builder: (context) => PreviewScreen(
            image: image,
          ),
        ),
      );
    } catch (e) {
      print(e);
    }
  }

  @override
  Widget build(BuildContext context) {
    if (!_isCameraPermissionGranted) {
      return const Center(child: Text('Camera permission not granted'));
    }

    return Scaffold(
      // Remove the AppBar to make camera full screen
      body: Stack(
        fit: StackFit.expand,
        children: [
          // Camera Preview that takes full screen
          FutureBuilder<void>(
            future: _initializeControllerFuture,
            builder: (context, snapshot) {
              if (snapshot.connectionState == ConnectionState.done) {
                return CameraPreview(_controller);
              } else {
                return const Center(child: CircularProgressIndicator());
              }
            },
          ),
          // Positioned capture button at bottom center
          Positioned(
            bottom: 30,
            left: 0,
            right: 0,
            child: Center(
              child: Container(
                height: 80,
                width: 80,
                child: TextButton(
                  onPressed: _takePicture,
                  style: TextButton.styleFrom(
                    shape: const CircleBorder(),
                    backgroundColor: Colors.white.withOpacity(0.8),
                    padding: const EdgeInsets.all(20),
                  ),
                  child: const Icon(
                    Icons.camera,
                    size: 36,
                    color: Colors.black,
                  ),
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }
}