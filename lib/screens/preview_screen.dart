import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import '../services/ocr_service.dart';
import '../models/ocr_result.dart';
import '../services/kvdb_service.dart';

class PreviewScreen extends StatefulWidget {
  final XFile image;

  const PreviewScreen({Key? key, required this.image}) : super(key: key);

  @override
  _PreviewScreenState createState() => _PreviewScreenState();
}

class _PreviewScreenState extends State<PreviewScreen> {
  final OCRService _ocrService = OCRService();
  final KVDBService _kvdbService = KVDBService();
  bool _isProcessing = true;
  String _errorMessage = '';
  List<Map<String, dynamic>>? _processedData;

  @override
  void initState() {
    super.initState();
    _processImage();
  }

  Future<void> _processImage() async {
    try {
      // Process image and get OCR results
      await _ocrService.loadModels();
      final results = await _ocrService.processImage(widget.image);
      
      // Write to KVDB
      final key = await _kvdbService.writeData(results);
      
      // Wait for processing
      await Future.delayed(const Duration(seconds: 8));
      
      // Read from KVDB
      final data = await _kvdbService.readData("1735902270721");
      
      // Extract only the Processed_data
      final processedData = (data['Processed_data'] as List?)
          ?.cast<Map<String, dynamic>>() ?? [];

      setState(() {
        _processedData = processedData;
        _isProcessing = false;
      });
    } catch (e) {
      setState(() {
        _errorMessage = e.toString();
        _isProcessing = false;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Extracted Data'),
      ),
      body: _buildBody(),
    );
  }

  Widget _buildBody() {
    if (_isProcessing) {
      return const Center(child: CircularProgressIndicator());
    }

    if (_errorMessage.isNotEmpty) {
      return Center(
        child: Text(_errorMessage, style: const TextStyle(color: Colors.red)),
      );
    }

    if (_processedData == null || _processedData!.isEmpty) {
      return const Center(child: Text('No processed data available'));
    }

    return SingleChildScrollView(
      scrollDirection: Axis.horizontal,
      child: SingleChildScrollView(
        child: Padding(
          padding: const EdgeInsets.all(16.0),
          child: DataTable(
            headingTextStyle: const TextStyle(
              fontWeight: FontWeight.bold,
              color: Colors.blue,
            ),
            columns: const [
              DataColumn(label: Text('Country')),
              DataColumn(label: Text('Exchange Rate')),
              DataColumn(label: Text('Service')),
              DataColumn(label: Text('Currency')),
              DataColumn(label: Text('Product')),
              DataColumn(label: Text('Price Range')),
            ],
            rows: _processedData!.map((data) => DataRow(
              cells: [
                DataCell(Text(data['Country'] ?? 'N/A')),
                DataCell(Text(data['Exchange Rate']?.toString() ?? 'N/A')),
                DataCell(Text(data['Service'] ?? 'N/A')),
                DataCell(Text(data['Currency'] ?? 'N/A')),
                DataCell(Text(data['Product'] ?? 'N/A')),
                DataCell(Text(data['PriceRange'] ?? 'N/A')),
              ],
            )).toList(),
          ),
        ),
      ),
    );
  }
}
