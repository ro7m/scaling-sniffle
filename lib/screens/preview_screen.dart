import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';
import 'dart:io';
import 'package:path_provider/path_provider.dart';

class PreviewScreen extends StatefulWidget {
  final XFile image;
  final String key;

  const PreviewScreen({Key? key, required this.image, required this.key}) : super(key: key);

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
    await Future.delayed(const Duration(seconds: 20)); // Wait for processing
    final response = await http.get(
      Uri.parse('https://kvdb.io/VuKUzo8aFSpoWpyXKpFxxH/${widget.key}'),
    );

    if (response.statusCode == 200) {
      return jsonDecode(response.body);
    } else {
      throw Exception('Failed to read data from upstream');
    }
  }

  Future<void> _downloadCsv(Map<String, dynamic> data) async {
    final keys = data.keys.toList();
    final values = keys.map((key) => data[key]).toList();

    final String csvData = '${keys.join(',')}\n${values.join(',')}';
    final directory = await getApplicationDocumentsDirectory();
    final path = '${directory.path}/data.csv';
    final File file = File(path);
    await file.writeAsString(csvData);

    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(content: Text('CSV downloaded to $path')),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Extracted Data'),
      ),
      body: FutureBuilder<Map<String, dynamic>>(
        future: _futureData,
        builder: (context, snapshot) {
          if (snapshot.connectionState == ConnectionState.waiting) {
            return Center(child: Column(
              mainAxisAlignment: MainAxisAlignment.center,
              children: const [
                CircularProgressIndicator(),
                SizedBox(height: 20),
                Text('Fetching processed data...')
              ],
            ));
          } else if (snapshot.hasError) {
            return Center(child: Text('Error: ${snapshot.error}'));
          } else if (!snapshot.hasData || snapshot.data == null) {
            return Center(child: Text('No data found'));
          }

          final data = snapshot.data!;
          return Column(
            children: [
              Image.file(File(widget.image.path)),
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
                        columns: data.keys.map((key) => DataColumn(label: Text(key))).toList(),
                        rows: [
                          DataRow(
                            cells: data.values.map((value) => DataCell(Text(value.toString()))).toList(),
                          ),
                        ],
                      ),
                    ),
                  ),
                ),
              ),
              ElevatedButton(
                onPressed: () => _downloadCsv(data),
                child: Text('Download as CSV'),
              ),
            ],
          );
        },
      ),
    );
  }
}