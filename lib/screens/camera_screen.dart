import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter/foundation.dart';
import 'package:camera/camera.dart';
import 'package:path_provider/path_provider.dart';
import 'package:path/path.dart' show join;
import 'package:permission_handler/permission_handler.dart';
import 'dart:convert';
import 'dart:io';
import 'package:http/http.dart' as http;

import 'preview_screen.dart';

class CornerEdgesPainter extends CustomPainter {
  final Color color;
  final double edgeSize;
  final double strokeWidth;

  CornerEdgesPainter({
    this.color = Colors.yellow,
    this.edgeSize = 40,
    this.strokeWidth = 4,
  });

  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..color = color
      ..strokeWidth = strokeWidth
      ..style = PaintingStyle.stroke;

    // Top Left Corner
    canvas.drawLine(
      Offset.zero,
      Offset(edgeSize, 0),
      paint,
    );
    canvas.drawLine(
      Offset.zero,
      Offset(0, edgeSize),
      paint,
    );

    // Top Right Corner
    canvas.drawLine(
      Offset(size.width, 0),
      Offset(size.width - edgeSize, 0),
      paint,
    );
    canvas.drawLine(
      Offset(size.width, 0),
      Offset(size.width, edgeSize),
      paint,
    );

    // Bottom Left Corner
    canvas.drawLine(
      Offset(0, size.height),
      Offset(edgeSize, size.height),
      paint,
    );
    canvas.drawLine(
      Offset(0, size.height),
      Offset(0, size.height - edgeSize),
      paint,
    );

    // Bottom Right Corner
    canvas.drawLine(
      Offset(size.width, size.height),
      Offset(size.width - edgeSize, size.height),
      paint,
    );
    canvas.drawLine(
      Offset(size.width, size.height),
      Offset(size.width, size.height - edgeSize),
      paint,
    );
  }

  @override
  bool shouldRepaint(CustomPainter oldDelegate) => false;
}

class CameraScreen extends StatefulWidget {
  final List<CameraDescription> cameras;

  const CameraScreen({Key? key, required this.cameras}) : super(key: key);

  @override
  CameraScreenState createState() => CameraScreenState();
}

class CameraScreenState extends State<CameraScreen> {
  late CameraController? _controller;
  late Future<void> _initializeControllerFuture;
  bool _isCameraPermissionGranted = false;
  double _minZoomLevel = 1.0;
  double _maxZoomLevel = 1.0;
  double _currentZoomLevel = 1.0;
  bool get _isSimulator => kDebugMode && Platform.isIOS;

  final List<String> _competitors = ['cp1', 'cp2', 'cp3', 'cp4', 'cp5'];
  final List<String> _countries = ['ES', 'FR', 'DE', 'IT', 'NL'];
  final Map<String, String> _countryCurrency = {
    'ES': 'EUR',
    'FR': 'EUR',
    'DE': 'EUR',
    'IT': 'EUR',
    'NL': 'EUR',
  };
  String? _selectedCompetitor;
  String? _selectedCountry;

  @override
  void initState() {
    super.initState();
    SystemChrome.setPreferredOrientations([
      DeviceOrientation.portraitUp,
      DeviceOrientation.portraitDown,
    ]);
    _checkSimulator();
  }

  Future<void> _checkSimulator() async {
    if (_isSimulator) {
      setState(() {
        _isCameraPermissionGranted = true;
      });
    } else {
      await _requestCameraPermission();
    }
  }

  @override
  void dispose() {
    SystemChrome.setPreferredOrientations([
      DeviceOrientation.portraitUp,
      DeviceOrientation.portraitDown,
    ]);
    _controller?.dispose();
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
    if (_isSimulator) return;

    _controller = CameraController(
      widget.cameras[0],
      ResolutionPreset.high,
      enableAudio: false,
    );

    _initializeControllerFuture = _controller!.initialize().then((_) async {
      if (!mounted) return;

      _minZoomLevel = await _controller!.getMinZoomLevel();
      _maxZoomLevel = await _controller!.getMaxZoomLevel();

      setState(() {});
    });
  }

  Future<XFile> _getSimulatorImage() async {
    final ByteData data = await rootBundle.load('assets/images/sample_document.png');
    final String path = join(
      (await getTemporaryDirectory()).path,
      'sample_document.png',
    );

    final File file = File(path);
    await file.writeAsBytes(data.buffer.asUint8List());

    return XFile(path);
  }

  Future<void> _takePicture() async {
    final timestamp = DateTime.now().millisecondsSinceEpoch.toString();

    try {
      if (_isSimulator) {
        final XFile mockImage = await _getSimulatorImage();
        if (!mounted) return;

        _sendDataInBackground(mockImage,timestamp);
        Navigator.of(context).pushReplacement(
          MaterialPageRoute(
            builder: (context) => PreviewScreen(
              msgkey: timestamp, // Pass an empty key initially
            ),
          ),
        );
        return;
      }

      await _initializeControllerFuture;
      final image = await _controller!.takePicture();

      if (!mounted) return;

      _sendDataInBackground(image,timestamp);
      Navigator.of(context).pushReplacement(
        MaterialPageRoute(
          builder: (context) => PreviewScreen(
            msgkey: timestamp, // Pass an empty key initially
          ),
        ),
      );
    } catch (e) {
      print(e);
    }
  }

  void _sendDataInBackground(XFile image, String timestamp) async {
    final String userId = ModalRoute.of(context)?.settings.arguments as String;
    final bytes = await image.readAsBytes();
    final String base64Image = base64Encode(bytes);
    final String? currency = _countryCurrency[_selectedCountry];
    final Map<String, dynamic> data = {
      'uploadedAt': timestamp,
      'agent_id': userId,
      'agent_account_name': 'locutor es 1',
      'competitor': _selectedCompetitor,
      'send_country': _selectedCountry,
      'send_currency': currency,
      'base64Image': base64Image
    };

    final response = await http.put(
      Uri.parse('https://kvdb.io/WkFjge45cAR9LKcWyrhAS4/$timestamp'),
      headers: {
        'Content-Type': 'application/json',
      },
      body: jsonEncode(data),
    );

    if (response.statusCode == 200) {
      // Update the key in the PreviewScreen once the data is sent successfully
      Navigator.of(context).pushReplacement(
        MaterialPageRoute(
          builder: (context) => PreviewScreen(
            msgkey: timestamp, // Pass the actual key
          ),
        ),
      );
    } else {
      print('Failed in upstream..');
    }
  }

  Future<void> _setZoomLevel(double value) async {
    if (_isSimulator) return;

    setState(() {
      _currentZoomLevel = value;
    });
    await _controller?.setZoomLevel(value);
  }

  Widget _buildCameraPreview() {
    if (_isSimulator) {
      return Image.asset(
        'assets/images/sample_document.png',
        fit: BoxFit.cover,
      );
    }

    return FutureBuilder<void>(
      future: _initializeControllerFuture,
      builder: (context, snapshot) {
        if (snapshot.connectionState == ConnectionState.done) {
          return CameraPreview(_controller!);
        } else {
          return const Center(child: CircularProgressIndicator());
        }
      },
    );
  }

  @override
  Widget build(BuildContext context) {
    if (!_isCameraPermissionGranted && !_isSimulator) {
      return const Center(child: Text('No Camera permission'));
    }

    final screenSize = MediaQuery.of(context).size;
    final cameraHeight = screenSize.height * 0.7;
    final cameraWidth = screenSize.width * 0.9;
    final String userId = ModalRoute.of(context)?.settings.arguments as String;

    return Scaffold(
      appBar: AppBar(
        title: Text(''),
        leading: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(Icons.person),
            SizedBox(width: 8),
            Text(userId),
          ],
        ),
        actions: [
          DropdownButton<String>(
            hint: Text('Competitor'),
            value: _selectedCompetitor,
            items: _competitors.map((String value) {
              return DropdownMenuItem<String>(
                value: value,
                child: Text(value),
              );
            }).toList(),
            onChanged: (String? newValue) {
              setState(() {
                _selectedCompetitor = newValue;
              });
            },
          ),
          SizedBox(width: 16),
          DropdownButton<String>(
            hint: Text('Country'),
            value: _selectedCountry,
            items: _countries.map((String value) {
              return DropdownMenuItem<String>(
                value: value,
                child: Text(value),
              );
            }).toList(),
            onChanged: (String? newValue) {
              setState(() {
                _selectedCountry = newValue;
              });
            },
          ),
        ],
      ),
      backgroundColor: Colors.black,
      body: Center(
        child: Stack(
          alignment: Alignment.center,
          children: [
            // Camera Preview or Simulator Image
            ClipRRect(
              borderRadius: BorderRadius.circular(12),
              child: SizedBox(
                width: cameraWidth,
                height: cameraHeight,
                child: FittedBox(
                  fit: BoxFit.cover,
                  child: SizedBox(
                    width: cameraWidth,
                    height: cameraHeight,
                    child: _buildCameraPreview(),
                  ),
                ),
              ),
            ),
            // Corner Edges Guide
            SizedBox(
              width: cameraWidth,
              height: cameraHeight,
              child: CustomPaint(
                painter: CornerEdgesPainter(
                  color: Colors.yellow,
                  edgeSize: 20,
                  strokeWidth: 2,
                ),
              ),
            ),

            // Zoom Controls
            Positioned(
              right: (screenSize.width - cameraWidth) / 2 + 16,
              top: (screenSize.height - cameraHeight) / 2 + 16,
              child: Container(
                decoration: BoxDecoration(
                  color: Colors.black54,
                  borderRadius: BorderRadius.circular(16),
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
              right: (screenSize.width - cameraWidth) / 2 + 16,
              top: (screenSize.height - cameraHeight) / 2 + 100,
              child: Container(
                padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                decoration: BoxDecoration(
                  color: Colors.black54,
                  borderRadius: BorderRadius.circular(8),
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

            // Guide Text
            Positioned(
              top: (screenSize.height - cameraHeight) / 2 - 40,
              child: Container(
                padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
                decoration: BoxDecoration(
                  color: Colors.black54,
                  borderRadius: BorderRadius.circular(20),
                ),
                child: const Text(
                  'Align document within the corners',
                  style: TextStyle(
                    color: Colors.white,
                    fontSize: 10,
                  ),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}