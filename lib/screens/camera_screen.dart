import 'package:flutter/material.dart';
import 'package:flutter/services.dart'; // For orientation lock
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
  double _minZoomLevel = 1.0;
  double _maxZoomLevel = 1.0;
  double _currentZoomLevel = 1.0;

  @override
  void initState() {
    super.initState();
    // Lock orientation to portrait
    SystemChrome.setPreferredOrientations([
      DeviceOrientation.portraitUp,
      DeviceOrientation.portraitDown,
    ]);
    _requestCameraPermission();
  }

  @override
  void dispose() {
    // Allow all orientations when leaving the camera screen
    SystemChrome.setPreferredOrientations([
      DeviceOrientation.portraitUp,
      DeviceOrientation.portraitDown,
      DeviceOrientation.landscapeLeft,
      DeviceOrientation.landscapeRight,
    ]);
    _controller.dispose();
    super.dispose();
  }

  Future<void> _requestCameraPermission() async {
    final status = await Permission.camera.request();
    setState(() {
      _isCameraPermissionGranted = status == PermissionStatus.granted;
    });
    if (_isCameraPermissionGranted) {
      await _initializeCamera();
    }
  }

  Future<void> _initializeCamera() async {
    _controller = CameraController(
      widget.cameras[0],
      ResolutionPreset.high,
      enableAudio: false,
    );
    
    _initializeControllerFuture = _controller.initialize().then((_) async {
      if (!mounted) return;
      
      // Get zoom range
      _minZoomLevel = await _controller.getMinZoomLevel();
      _maxZoomLevel = await _controller.getMaxZoomLevel();
      
      setState(() {});
    });
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

  Future<void> _setZoomLevel(double value) async {
    setState(() {
      _currentZoomLevel = value;
    });
    await _controller.setZoomLevel(value);
  }

  @override
  Widget build(BuildContext context) {
    if (!_isCameraPermissionGranted) {
      return const Center(child: Text('Camera permission not granted'));
    }

    return Scaffold(
      body: Stack(
        fit: StackFit.expand,
        children: [
          // Camera Preview
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

          // Rectangular Guide Box
          Center(
            child: Container(
              width: MediaQuery.of(context).size.width * 0.8,
              height: MediaQuery.of(context).size.width * 0.3, // Aspect ratio for document
              decoration: BoxDecoration(
                border: Border.all(
                  color: Colors.white,
                  width: 2.0,
                ),
                borderRadius: BorderRadius.circular(8),
              ),
            ),
          ),

          // Zoom Controls
          Positioned(
            right: 16,
            top: MediaQuery.of(context).padding.top + 16,
            child: Container(
              decoration: BoxDecoration(
                color: Colors.black54,
                borderRadius: BorderRadius.circular(20),
              ),
              padding: const EdgeInsets.symmetric(vertical: 8),
              child: RotatedBox(
                quarterTurns: 3,
                child: Slider(
                  value: _currentZoomLevel,
                  min: _minZoomLevel,
                  max: _maxZoomLevel,
                  activeColor: Colors.white,
                  inactiveColor: Colors.white30,
                  onChanged: (value) => _setZoomLevel(value),
                ),
              ),
            ),
          ),

          // Zoom Level Indicator
          Positioned(
            right: 16,
            top: MediaQuery.of(context).padding.top + 100,
            child: Container(
              padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
              decoration: BoxDecoration(
                color: Colors.black54,
                borderRadius: BorderRadius.circular(12),
              ),
              child: Text(
                '${_currentZoomLevel.toStringAsFixed(1)}x',
                style: const TextStyle(color: Colors.white),
              ),
            ),
          ),

          // Capture Button
          Positioned(
            bottom: 30,
            left: 0,
            right: 0,
            child: Center(
              child: Container(
                height: 80,
                width: 80,
                decoration: BoxDecoration(
                  shape: BoxShape.circle,
                  border: Border.all(color: Colors.white, width: 4),
                ),
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

          // Guide Text
          Positioned(
            top: MediaQuery.of(context).padding.top + 200,
            left: 0,
            right: 0,
            child: Center(
              child: Container(
                padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
                decoration: BoxDecoration(
                  color: Colors.black54,
                  borderRadius: BorderRadius.circular(20),
                ),
                child: const Text(
                  'Align document within the box',
                  style: TextStyle(
                    color: Colors.white,
                    fontSize: 16,
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