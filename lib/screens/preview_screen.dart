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

  @override
  void initState() {
    super.initState();
    _modelLoadingFuture = _ocrService.loadModels();
    _processImage();
  }

  Future<void> _processImage() async {
    setState(() {
      _isProcessing = true;
      _error = '';
    });

    try {
      await _modelLoadingFuture;
      
      final bytes = await File(widget.imagePath).readAsBytes();
      final codec = await ui.instantiateImageCodec(bytes);
      final frameInfo = await codec.getNextFrame();
      final image = frameInfo.image;

      final results = await _ocrService.processImage(image);
      
      setState(() {
        _results = results;
        _isProcessing = false;
      });
    } catch (e) {
      setState(() {
        _error = 'Error processing image: $e';
        _isProcessing = false;
      });
    }
  }

  void _handleRetry() {
    Navigator.of(context).pop(_results);
  }

  void _handleAccept() {
    Navigator.of(context).pop(_results);
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Preview & OCR Result')),
      body: Column(
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
    );
  }

  Widget _buildResultsWidget() {
    if (_isProcessing) {
      return const Center(
        child: CircularProgressIndicator(),
      );
    }

    if (_error.isNotEmpty) {
      return Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Text(_error, style: const TextStyle(color: Colors.red)),
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
          child: ListView.builder(
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