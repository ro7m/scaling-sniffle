import 'dart:io';
import 'package:flutter/material.dart';
import '../services/ocr_service.dart';

class PreviewScreen extends StatefulWidget {
  final String imagePath;

  const PreviewScreen({Key? key, required this.imagePath}) : super(key: key);

  @override
  PreviewScreenState createState() => PreviewScreenState();
}

class PreviewScreenState extends State<PreviewScreen> {
  late Future<String> _extractedText;
  final OCRService _ocrService = OCRService();

  @override
  void initState() {
    super.initState();
    _extractedText = _ocrService.extractText(widget.imagePath);
  }

  void _retryOCR() {
    setState(() {
      _extractedText = _ocrService.extractText(widget.imagePath);
    });
  }

  void _acceptResult() {
    // Handle the accepted result (e.g., save to database, navigate back, etc.)
    Navigator.of(context).pop();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Preview & OCR Result')),
      body: Column(
        children: [
          Expanded(
            flex: 1,
            child: Image.file(
              File(widget.imagePath),
              fit: BoxFit.contain,
            ),
          ),
          Expanded(
            flex: 1,
            child: Padding(
              padding: const EdgeInsets.all(16.0),
              child: FutureBuilder<String>(
                future: _extractedText,
                builder: (context, snapshot) {
                  if (snapshot.connectionState == ConnectionState.waiting) {
                    return const Center(child: CircularProgressIndicator());
                  } else if (snapshot.hasError) {
                    return Center(child: Text('Error: ${snapshot.error}'));
                  } else {
                    return Column(
                      children: [
                        Expanded(
                          child: SingleChildScrollView(
                            child: Text(
                              snapshot.data ?? 'No text extracted',
                              style: const TextStyle(fontSize: 18),
                            ),
                          ),
                        ),
                        Row(
                          mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                          children: [
                            ElevatedButton(
                              onPressed: _retryOCR,
                              child: const Text('Retry'),
                            ),
                            ElevatedButton(
                              onPressed: _acceptResult,
                              child: const Text('Accept'),
                            ),
                          ],
                        ),
                      ],
                    );
                  }
                },
              ),
            ),
          ),
        ],
      ),
    );
  }
}
