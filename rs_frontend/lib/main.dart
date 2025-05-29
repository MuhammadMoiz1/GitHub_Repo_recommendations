import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:url_launcher/url_launcher.dart';

void main() {
  runApp(RecommendationApp());
}

class RecommendationApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'GitHub Recommender',
      theme: ThemeData(
        primarySwatch: Colors.teal,
        visualDensity: VisualDensity.adaptivePlatformDensity,
      ),
      home: RecommendationPage(),
      debugShowCheckedModeBanner: false,
    );
  }
}

class RecommendationPage extends StatefulWidget {
  @override
  _RecommendationPageState createState() => _RecommendationPageState();
}

class _RecommendationPageState extends State<RecommendationPage> {
  final TextEditingController _tokenController = TextEditingController();
  bool _loading = false;
  List<dynamic> _recommendations = [];

  Future<void> _fetchRecommendations() async {
    final String token = _tokenController.text.trim();
    if (token.isEmpty) {
      ScaffoldMessenger.of(
        context,
      ).showSnackBar(SnackBar(content: Text('Please enter a GitHub token')));
      return;
    }

    setState(() {
      _loading = true;
      _recommendations.clear();
    });

    final url = Uri.parse(
      'http://127.0.0.1:8000/recommend',
    ); // Update if deployed

    try {
      final response = await http.post(
        url,
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({'token': token}),
      );

      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        setState(() {
          _recommendations = data['recommendations']['recommendations'];
        });
      } else {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Error: ${response.reasonPhrase}')),
        );
      }
    } catch (e) {
      ScaffoldMessenger.of(
        context,
      ).showSnackBar(SnackBar(content: Text('Failed to connect to the API')));
    }

    setState(() {
      _loading = false;
    });
  }

  Widget _buildRecommendationCard(dynamic item) {
    return Card(
      margin: EdgeInsets.symmetric(vertical: 8),
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
      elevation: 4,
      child: ListTile(
        title: Text(
          item['name'],
          style: TextStyle(fontWeight: FontWeight.bold),
        ),
        subtitle: Padding(
          padding: const EdgeInsets.only(top: 6.0),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(item['description'] ?? 'No description'),
              SizedBox(height: 8),
              Wrap(
                spacing: 10,
                children: [
                  if (item['language'] != null)
                    Chip(
                      label: Text(item['language']),
                      backgroundColor: Colors.teal.shade50,
                    ),
                  Text("Score: ${item['score'].toStringAsFixed(4)}"),
                ],
              ),
            ],
          ),
        ),
        trailing: Icon(Icons.open_in_new),
        onTap: () => launchURL(item['url']),
      ),
    );
  }

  Future<void> launchURL(String url) async {
    final Uri uri = Uri.parse(url);
    if (await canLaunchUrl(uri)) {
      await launchUrl(uri, mode: LaunchMode.externalApplication);
    } else {
      ScaffoldMessenger.of(
        context,
      ).showSnackBar(SnackBar(content: Text('Could not open the link')));
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text('GitHub Repo Recommender'), centerTitle: true),
      body: Center(
        child: Container(
          constraints: BoxConstraints(maxWidth: 800),
          padding: EdgeInsets.all(24),
          child: Column(
            children: [
              TextField(
                controller: _tokenController,
                decoration: InputDecoration(
                  labelText: 'Enter your GitHub token',
                  border: OutlineInputBorder(),
                ),
              ),
              SizedBox(height: 16),
              ElevatedButton.icon(
                icon: Icon(Icons.send),
                label: Text('Get Recommendations'),
                onPressed: _fetchRecommendations,
              ),
              SizedBox(height: 30),
              _loading
                  ? CircularProgressIndicator()
                  : Expanded(
                    child: ListView.builder(
                      itemCount: _recommendations.length,
                      itemBuilder: (context, index) {
                        return _buildRecommendationCard(
                          _recommendations[index],
                        );
                      },
                    ),
                  ),
            ],
          ),
        ),
      ),
    );
  }
}
