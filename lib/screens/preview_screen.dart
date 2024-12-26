import 'dart:io';
import 'dart:ui' as ui;
import 'package:flutter/material.dart';
import '../services/ocr_service.dart';
import '../models/ocr_result.dart';
import '../widgets/bounding_box_painter.dart';

class PreviewScreen extends StatefulWidget {
  final String imagePath;

  const PreviewScreen({Key? key, required this.imagePath}) : super(key: key);

  @override
  PreviewScreenState createState() => PreviewScreenState();
}

class PreviewScreenState extends State<PreviewScreen> {
  final OCRService _ocrService = OCRService();
  late Future<void> _modelLoadingFuture;
  List<OCRResult> _results = [];
  bool _isProcessing = false;
  String _error = '';
  List<String> _debugMessages = [];
  bool _showDebug = true;

  @override
  void initState() {
    super.initState();
    _ocrService.debugCallback = _addDebugMessage;
    _modelLoadingFuture = _ocrService.loadModels();
    _processImage();
  }

  void _addDebugMessage(String message) {
    print(message); // Still print to console
    setState(() {
      _debugMessages.add("[${DateTime.now().toString().split('.')[0]}] $message");
      // Keep only last 50 messages to prevent memory issues
      if (_debugMessages.length > 50) {
        _debugMessages.removeAt(0);
      }
    });
  }

  Future<void> _processImage() async {
    setState(() {
      _isProcessing = true;
      _error = '';
      _debugMessages.clear(); // Clear previous messages
    });

    try {
      _addDebugMessage('Starting OCR process...');
      _addDebugMessage('Loading models...');
      await _modelLoadingFuture;
      _addDebugMessage('Models loaded successfully');
      
      _addDebugMessage('Reading image file...');
      final bytes = await File(widget.imagePath).readAsBytes();
      final codec = await ui.instantiateImageCodec(bytes);
      final frameInfo = await codec.getNextFrame();
      final image = frameInfo.image;
      _addDebugMessage('Image loaded: ${image.width}x${image.height}');

      _addDebugMessage('Processing image with OCR...');
      final results = await _ocrService.processImage(image);
      _addDebugMessage('OCR Results: ${results.length} items found');
      
      if (results.isEmpty) {
        _addDebugMessage('No text detected in the image');
      } else {
        results.forEach((result) {
          _addDebugMessage('Found text: "${result.text}" at (${result.boundingBox.x.toStringAsFixed(1)}, ${result.boundingBox.y.toStringAsFixed(1)})');
        });
      }

      setState(() {
        _results = results;
        _isProcessing = false;
      });
    } catch (e, stackTrace) {
      _addDebugMessage('Error: $e');
      _addDebugMessage('Stack trace: $stackTrace');
      setState(() {
        _error = 'Error processing image: $e';
        _isProcessing = false;
      });
    }
  }

  void _handleRetry() {
    _processImage();
  }

  void _handleAccept() {
    Navigator.of(context).pop(_results);
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Preview & OCR Result'),
        actions: [
          IconButton(
            icon: Icon(_showDebug ? Icons.bug_report : Icons.bug_report_outlined),
            onPressed: () {
              setState(() {
                _showDebug = !_showDebug;
              });
            },
          ),
        ],
      ),
      body: Stack(
        children: [
          Column(
            children: [
              Expanded(
                flex: 1,
                child: Stack(
                  children: [
                    Image.file(
                      File(widget.imagePath),
                      fit: BoxFit.contain,
                    ),
                    CustomPaint(
                      painter: BoundingBoxPainter(_results),
                      size: Size.infinite,
                    ),
                  ],
                ),
              ),
              Expanded(
                flex: 1,
                child: Padding(
                  padding: const EdgeInsets.all(16.0),
                  child: _buildResultsWidget(),
                ),
              ),
            ],
          ),
          // Debug overlay
          if (_showDebug)
            Positioned(
              top: 0,
              left: 0,
              right: 0,
              child: Container(
                color: Colors.black.withOpacity(0.7),
                height: 200,
                child: ListView.builder(
                  padding: const EdgeInsets.all(8),
                  itemCount: _debugMessages.length,
                  itemBuilder: (context, index) {
                    return Text(
                      _debugMessages[index],
                      style: const TextStyle(
                        color: Colors.white,
                        fontSize: 12,
                      ),
                    );
                  },
                ),
              ),
            ),
        ],
      ),
    );
  }

  Widget _buildResultsWidget() {
    if (_isProcessing) {
      return const Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            CircularProgressIndicator(),
            SizedBox(height: 16),
            Text('Processing image...'),
          ],
        ),
      );
    }

    if (_error.isNotEmpty) {
      return Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Text(_error, 
              style: const TextStyle(color: Colors.red),
              textAlign: TextAlign.center,
            ),
            const SizedBox(height: 16),
            ElevatedButton(
              onPressed: _handleRetry,
              child: const Text('Retry'),
            ),
          ],
        ),
      );
    }

    return Column(
      children: [
        Expanded(
          child: _results.isEmpty
              ? const Center(
                  child: Text('No text detected'),
                )
              : ListView.builder(
                  itemCount: _results.length,
                  itemBuilder: (context, index) {
                    return ListTile(
                      title: Text(_results[index].text),
                      subtitle: Text(
                        'Box: (${_results[index].boundingBox.x.toStringAsFixed(1)}, '
                        '${_results[index].boundingBox.y.toStringAsFixed(1)})',
                      ),
                    );
                  },
                ),
        ),
        Row(
          mainAxisAlignment: MainAxisAlignment.spaceEvenly,
          children: [
            ElevatedButton(
              onPressed: _handleRetry,
              child: const Text('Retry'),
            ),
            ElevatedButton(
              onPressed: _handleAccept,
              child: const Text('Accept'),
            ),
          ],
        ),
      ],
    );
  }
}