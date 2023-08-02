import 'dart:io';
import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:tflite/tflite.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:image/image.dart' as img;
import 'package:camera/camera.dart';

void main() => runApp(SkinCancerDetectionApp());

class SkinCancerDetectionApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      home: SkinCancerDetectionScreen(),
    );
  }
}

class SkinCancerDetectionScreen extends StatefulWidget {
  @override
  _SkinCancerDetectionScreenState createState() =>
      _SkinCancerDetectionScreenState();
}

class _SkinCancerDetectionScreenState extends State<SkinCancerDetectionScreen> {
  List<String> labels = [
    'akiec',
    'bcc',
    'bkl',
    'df',
    'mel',
    'nv',
    'vasc',
  ];

  File? _image;
  bool _isLoading = false;
  String _result = '';

  final ImagePicker _picker = ImagePicker();
  late CameraController _cameraController;
  bool _isCameraInitialized = false;

  @override
  void initState() {
    super.initState();
    // Load the model
    loadModel();
    // Initialize the camera
    initCamera();
  }

  loadModel() async {
    try {
      String modelPath = 'assets/model_unquant1.tflite';
      await Tflite.loadModel(
        model: modelPath,
        labels: 'assets/labels1.txt',
      );
    } catch (e) {
      print('Error loading model: $e');
    }
  }

  initCamera() async {
    try {
      // Get a list of available cameras
      List<CameraDescription> cameras = await availableCameras();
      if (cameras.isEmpty) {
        return;
      }
      // Initialize the camera controller with the first available camera
      _cameraController = CameraController(cameras[0], ResolutionPreset.medium);
      await _cameraController.initialize();
      setState(() {
        _isCameraInitialized = true;
      });
    } catch (e) {
      print('Error initializing camera: $e');
    }
  }


  classifyImage(File image) async {
    if (image == null) return;

    setState(() {
      _isLoading = true;
    });

    try {
      // Resize and preprocess the image
      var bytes = await image.readAsBytes();
      img.Image? rawImage = img.decodeImage(bytes);
      img.Image resizedImage = img.copyResize(
          rawImage!, width: 224, height: 224);

      // Print resized image shape
      print('Resized image shape: ${resizedImage.width}x${resizedImage
          .height}x${getImageChannels(resizedImage)}');

      var input = ImageProcessorUtils.float32ListFromImage(resizedImage);

      // Print input tensor shape
      print('Input tensor shape: ${input.length}');

      // Run inference
      var output = await Tflite.runModelOnBinary(
        binary: input.buffer.asUint8List(),
      );

      if (output == null || output.isEmpty) {
        throw Exception('Empty output');
      }

      // Convert the output to a List<Map<String, dynamic>>
      List<Map<String, dynamic>> predictionList = [];
      for (var value in output) {
        predictionList.add(Map<String, dynamic>.from(value));
      }

      // Sort the predictions by confidence in descending order
      predictionList.sort((a, b) => b['confidence'].compareTo(a['confidence']));

      // Get the top prediction
      Map<String, dynamic> topPrediction = predictionList.first;

      // Get the predicted label and confidence
      String predictedLabel = labels[topPrediction['index']];
      double confidence = topPrediction['confidence'];

      setState(() {
        _result = '$predictedLabel (${(confidence * 100).toStringAsFixed(
            2)}% confidence)';
        _isLoading = false;
      });
    } catch (e) {
      print('Error classifying image: $e');
      setState(() {
        _isLoading = false;
        _result = 'Error';
      });
    }
  }


  int getImageChannels(img.Image image) {
    final channels = image
        .getBytes()
        .length ~/ (image.width * image.height);
    return channels;
  }


  pickImage() async {
    XFile? image = await _picker.pickImage(source: ImageSource.gallery);
    if (image != null) {
      setState(() {
        _image = File(image.path);
        _result = '';
      });
      classifyImage(_image!);
    }
  }

  pickImageFromCamera() async {
    XFile? image = await _picker.pickImage(source: ImageSource.camera);
    if (image != null) {
      setState(() {
        _image = File(image.path);
        _result = '';
      });
      classifyImage(_image!);
    }
  }

  void _showReadme() {
    showDialog(
      context: context,
      builder: (BuildContext context) {
        return AlertDialog(
          title: Text('Readme'),
          content: SingleChildScrollView(
            child: Text(
              'This demo skin cancer detection Android app project is completed by:\n\n'
                  '1. Aman Prakash Kanth\n'
                  '2. Giriraj Garg\n'
                  '3. Ashutosh Kumar\n'
                  '4. Shubh Raj\n\n'
                  'All the full forms of skin cancer types are:\n\n'
                  'akiec: Actinic Keratoses and Intraepidermal Carcinoma\n'
                  'bcc: Basal Cell Carcinoma\n'
                  'bkl: Benign Keratosis-like Lesions\n'
                  'df: Dermatofibroma\n'
                  'mel: Melanoma\n'
                  'nv: Melanocytic Nevi (benign moles)\n'
                  'vasc: Vascular Lesions\n',
            ),
          ),
          actions: [
            TextButton(
              onPressed: () {
                Navigator.of(context).pop();
              },
              child: Text('Close'),
            ),
          ],
        );
      },
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Skin Cancer Detection'),
        actions: [
          IconButton(
            onPressed: _showReadme,
            icon: Icon(Icons.info_outline),
          ),
        ],
      ),
      body: Center(
        child: _isLoading
            ? CircularProgressIndicator()
            : Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            if (_image != null) ...[
              Image.file(_image!, height: 200),
              SizedBox(height: 20),
            ],
            Text(_result, style: TextStyle(fontSize: 24)),
            SizedBox(height: 20),
            ElevatedButton(
              onPressed: pickImage,
              child: Text('Pick Image from Gallery'),
            ),
            SizedBox(height: 10),
            if (_isCameraInitialized) ...[
              ElevatedButton(
                onPressed: pickImageFromCamera,
                child: Text('Take a Photo'),
              ),
            ],
          ],
        ),
      ),
    );
  }


  @override
  void dispose() {
    // Dispose of the camera controller and Tflite model
    _cameraController.dispose();
    Tflite.close();
    super.dispose();
  }
}

class ImageProcessorUtils {
  static Float32List float32ListFromImage(img.Image resizedImage) {
    // Convert the resized image to a 1D Float32List (float32) for TensorFlow Lite model
    var convertedBytes = Float32List(224 * 224 * 3);
    int pixelIndex = 0;
    var bytes = resizedImage.getBytes();
    for (int i = 0; i < bytes.length; i += 3) {
      convertedBytes[pixelIndex++] = bytes[i] / 255.0; // Normalize the Red channel (0-255) to (0-1)
      convertedBytes[pixelIndex++] = bytes[i + 1] / 255.0; // Normalize the Green channel (0-255) to (0-1)
      convertedBytes[pixelIndex++] = bytes[i + 2] / 255.0; // Normalize the Blue channel (0-255) to (0-1)
    }
    return convertedBytes;
  }
}






