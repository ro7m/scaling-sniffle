import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';
import 'dart:io';
import 'package:path_provider/path_provider.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:external_path/external_path.dart';

class PreviewScreen extends StatefulWidget {
  final String? msgkey;

  const PreviewScreen({Key? key, required this.msgkey}) : super(key: key);

  @override
  _PreviewScreenState createState() => _PreviewScreenState();
}

class _PreviewScreenState extends State<PreviewScreen> {
  late Future<Map<String, dynamic>> _futureData;

  @override
  void initState() {
    super.initState();
    _futureData = _fetchData();
  }

  Future<Map<String, dynamic>> _fetchData() async {
    try {
      await Future.delayed(const Duration(seconds: 25));
      final response = await http.get(
        Uri.parse('https://kvdb.io/VuKUzo8aFSpoWpyXKpFxxH/${widget.msgkey}'),
      );

      if (response.statusCode == 200) {
        return jsonDecode(response.body);
      } else {
        throw Exception('Failed to read data from upstream');
      }
    } catch (e) {
      print('Exception occurred while fetching data for msgkey: ${widget.msgkey}');
      print(e);
      rethrow;
    }
  }

  List<String> _getColumns(Map<String, dynamic> data) {
    if (data['Processed_data'] == null || 
        data['Processed_data'].isEmpty ||
        !(data['Processed_data'] is List)) {
      return [];
    }

    Set<String> columns = {'Key'};
    for (var item in data['Processed_data']) {
      if (item is Map<String, dynamic>) {
        columns.addAll(item.keys);
      }
    }
    return columns.toList();
  }

  List<Map<String, String>> _getRows(Map<String, dynamic> data) {
    if (data['Processed_data'] == null || 
        data['Processed_data'].isEmpty ||
        !(data['Processed_data'] is List)) {
      return [];
    }

    List<Map<String, String>> rows = [];
    final uploadedAt = data['Key']?.toString() ?? '';
    
    for (var item in data['Processed_data']) {
      if (item is Map<String, dynamic>) {
        Map<String, String> row = {'Key': uploadedAt};
        item.forEach((key, value) {
          row[key] = value?.toString() ?? '';
        });
        rows.add(row);
      }
    }
    return rows;
  }

  Future<void> _downloadCsv(Map<String, dynamic> data) async {
    try {
      // Request storage permission
      var status = await Permission.storage.request();
      if (!status.isGranted) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(content: Text('Storage permission is required to download the file')),
        );
        return;
      }

      final columns = _getColumns(data);
      final rows = _getRows(data);
      
      if (columns.isEmpty || rows.isEmpty) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(content: Text('No data to export')),
        );
        return;
      }

      final csvHeader = columns.join(',');
      final csvRows = rows.map((row) {
        return columns.map((col) => '"${row[col] ?? ''}"').join(',');
      }).join('\n');

      final csvContent = '$csvHeader\n$csvRows';
      
      // Get the download directory path based on platform
      String? downloadPath;
      if (Platform.isAndroid) {
        downloadPath = await ExternalPath.getExternalStoragePublicDirectory(
          ExternalPath.DIRECTORY_DOWNLOADS
        );
      } else if (Platform.isIOS) {
        final directory = await getApplicationDocumentsDirectory();
        downloadPath = directory.path;
      }

      if (downloadPath == null) {
        throw Exception('Could not determine download path');
      }

      // Use msgkey in filename
      final filename = 'data_${widget.msgkey}.csv';
      final path = '$downloadPath/$filename';
      
      final File file = File(path);
      await file.writeAsString(csvContent);

      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text('CSV downloaded to: $path'),
          duration: const Duration(seconds: 5),
        ),
      );
    } catch (e) {
      print('Error downloading CSV: $e');
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Error downloading CSV: $e')),
      );
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Data Preview'),
      ),
      body: FutureBuilder<Map<String, dynamic>>(
        future: _futureData,
        builder: (context, snapshot) {
          if (snapshot.connectionState == ConnectionState.waiting) {
            return Center(
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: const [
                  CircularProgressIndicator(),
                  SizedBox(height: 20),
                  Text('Processing the data now...'),
                ],
              ),
            );
          } else if (snapshot.hasError) {
            return Center(child: Text('Error: ${snapshot.error}'));
          } else if (!snapshot.hasData || snapshot.data == null) {
            return const Center(child: Text('No data found'));
          }

          final data = snapshot.data!;
          final columns = _getColumns(data);
          final rows = _getRows(data);

          if (columns.isEmpty || rows.isEmpty) {
            return const Center(child: Text('No data available'));
          }

          return Column(
            children: [
              Expanded(
                child: SingleChildScrollView(
                  scrollDirection: Axis.horizontal,
                  child: SingleChildScrollView(
                    child: Padding(
                      padding: const EdgeInsets.all(16.0),
                      child: DataTable(
                        headingTextStyle: const TextStyle(
                          fontWeight: FontWeight.bold,
                          color: Colors.blue,
                        ),
                        columns: columns
                            .map((col) => DataColumn(label: Text(col)))
                            .toList(),
                        rows: rows.map((row) {
                          return DataRow(
                            cells: columns
                                .map((col) => DataCell(Text(row[col] ?? '')))
                                .toList(),
                          );
                        }).toList(),
                      ),
                    ),
                  ),
                ),
              ),
              Padding(
                padding: const EdgeInsets.all(16.0),
                child: SizedBox(
                  width: double.infinity,
                  height: 50.0,
                  child: ElevatedButton(
                    onPressed: () => _downloadCsv(data),
                    child: const Text('Download as CSV'),
                  ),
                ),
              ),
            ],
          );
        },
      ),
    );
  }
}
