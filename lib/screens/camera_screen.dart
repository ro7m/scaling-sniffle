import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'preview_screen.dart'; // Import the PreviewScreen

class CameraScreen extends StatefulWidget {
  @override
  _CameraScreenState createState() => _CameraScreenState();
}

class _CameraScreenState extends State<CameraScreen> {
  CameraController? _controller;
  List<CameraDescription>? cameras;
  CameraDescription? firstCamera;

  @override
  void initState() {
    super.initState();
    _initializeCamera();
  }

  Future<void> _initializeCamera() async {
    cameras = await availableCameras();
    firstCamera = cameras?.first;
    _controller = CameraController(
      firstCamera!,
      ResolutionPreset.high,
    );
    await _controller?.initialize();
    setState(() {});
  }

  @override
  Widget build(BuildContext context) {
    if (_controller == null || !_controller!.value.isInitialized) {
      return Center(child: CircularProgressIndicator());
    }

    return Scaffold(
      appBar: AppBar(title: Text("Camera")),
      body: Column(
        children: [
          Expanded(
            child: CameraPreview(_controller!),
          ),
          ElevatedButton(
            onPressed: () async {
              try {
                final image = await _controller?.takePicture();
                if (image != null) {
                  Navigator.push(
                    context,
                    MaterialPageRoute(
                      builder: (context) => PreviewScreen(image: image), // Pass the image file to the PreviewScreen
                    ),
                  );
                }
              } catch (e) {
                print(e);
              }
            },
            child: Text("Capture Image"),
          ),
        ],
      ),
    );
  }

  @override
  void dispose() {
    _controller?.dispose();
    super.dispose();
  }
}