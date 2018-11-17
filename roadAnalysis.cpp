

#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>

using namespace std;
using namespace cv;

/*static void help()
{
	cout << "\nThis program demonstrates the use of cv::CascadeClassifier class to detect objects (Face + eyes). You can use Haar or LBP features.\n"
		"This classifier can recognize many kinds of rigid objects, once the appropriate classifier is trained.\n"
		"It's most known use is for faces.\n"
		"Usage:\n"	
		"./facedetect [--cascade=<cascade_path> this is the primary trained classifier such as frontal face]\n"
		"   [--nested-cascade[=nested_cascade_path this an optional secondary classifier such as eyes]]\n"
		"   [--scale=<image scale greater or equal to 1, try 1.3 for example>]\n"
		"   [--try-flip]\n"
		"   [filename|camera_index]\n\n"
		"see facedetect.cmd for one call:\n"
		"./facedetect --cascade=\"../../data/haarcascades/haarcascade_frontalface_alt.xml\" --nested-cascade=\"../../data/haarcascades/haarcascade_eye_tree_eyeglasses.xml\" --scale=1.3\n\n"
		"During execution:\n\tHit any key to quit.\n"
		"\tUsing OpenCV version " << CV_VERSION << "\n" << endl;
}*/

void detectAndDraw(Mat& img, CascadeClassifier& cascade,
	CascadeClassifier& nestedCascade,
	double scale, bool tryflip);
 
void carDetectAndDraw(Mat& img, CascadeClassifier& cascade,double scale);

void pedestrianDetectAndDraw(Mat& img, CascadeClassifier& cascade, double scale);

string cascadeName, cascadeName2;
string nestedCascadeName;

int main(int argc, const char** argv)
{
	VideoCapture capture;
	Mat frame, image;
	string inputName;
	bool tryflip;
	CascadeClassifier carCascade,pedestrianCascade, nestedCascade;
	double scale;

	cv::CommandLineParser parser(argc, argv,
		"{help h||}" 
		"{carsCascade |cars|}"
		"{pedestriansCascade|haarcascade_pedestrians|}"
		"{nested-cascade|upper_bodyy.xml|}"
		"{scale|1|}{try-flip||}{@filename||}"
	);
	parser.printMessage();
	/*if (parser.has("help"))
	{
		help();
		return 0;
	}*/
	cascadeName = parser.get<string>("carsCascade") +		 ".xml";
	cascadeName2 = parser.get<string>("pedestriansCascade") + ".xml";

	cout << cascadeName;
	cout << cascadeName2;

	nestedCascadeName = parser.get<string>("nested-cascade");
	scale = 1.5;// parser.get<double>("scale");
	if (scale < 1)
		scale = 1;
	tryflip = parser.has("try-flip");
	inputName = parser.get<string>("@filename");
	if (!parser.check())
	{
		parser.printErrors();
		return 0;
	}
	if (!nestedCascade.load(nestedCascadeName))
		cerr << "WARNING: Could not load classifier cascade for nested objects" << endl;
	if (!carCascade.load(cascadeName) || !pedestrianCascade.load(cascadeName2))
	{
		cerr << "ERROR: Could not load classifier cascade" << endl;
		//help();
		return -1;
	}
	if (inputName.empty() || (isdigit(inputName[0]) && inputName.size() == 1))
	{
		int camera = inputName.empty() ? 0 : inputName[0] - '0';
		if (!capture.open(camera))
			cout << "Capture from camera #" << camera << " didn't work" << endl;
	}
	

	if (capture.isOpened())
	{
		cout << "Video capturing has been started ..." << endl;

		for (;;)
		{
			capture >> frame;
			if (frame.empty())
				break;

			Mat frame1 = frame.clone();
			carDetectAndDraw(frame1, carCascade, scale);
			pedestrianDetectAndDraw(frame1, pedestrianCascade, scale);
			//detectAndDraw(frame1, carCascade, nestedCascade, scale, tryflip);
			//detectAndDraw(frame1, pedestrianCascade, nestedCascade, scale, tryflip);


			char c = (char)waitKey(10);
			if (c == 27 || c == 'q' || c == 'Q')
				break;
		}
	}


	return 0;
}

void detectAndDraw(Mat& img, CascadeClassifier& cascade,
	CascadeClassifier& nestedCascade,
	double scale, bool tryflip)
{
	double t = 0;
	vector<Rect> faces, faces2;  // les rectangles qui contiendront les visages
	const static Scalar colors[] =
	{
		Scalar(255,0,0),
		Scalar(255,128,0),
		Scalar(255,255,0),
		Scalar(0,255,0),
		Scalar(0,128,255),
		Scalar(0,255,255),
		Scalar(0,0,255),
		Scalar(255,0,255)
	};
	Mat gray, smallImg;

	cvtColor(img, gray, COLOR_BGR2GRAY);
	double fx = 1 / scale;
	resize(gray, smallImg, Size(), fx, fx, INTER_LINEAR_EXACT);	
	equalizeHist(smallImg, smallImg);

	t = (double)getTickCount();
	cascade.detectMultiScale(smallImg, faces,
		1.01, 6, 0 
		//|CASCADE_FIND_BIGGEST_OBJECT,
		//|CASCADE_DO_ROUGH_SEARCH,
		//|CASCADE_SCALE_IMAGE,
		| CV_HAAR_DO_CANNY_PRUNING,
		Size(30, 30));
	if (tryflip)				// 
	{
		flip(smallImg, smallImg, 1);
		/*scaleFactor – Parameter specifying how much the image size is reduced at each image scale.

		Basically the scale factor is used to create your scale pyramid. More explanation can be found here. In short, as described here, your model has a fixed size defined during training, which is visible in the xml. This means that this size of face is detected in the image if present. However, by rescaling the input image, you can resize a larger face to a smaller one, making it detectable by the algorithm.

		1.05 is a good possible value for this, which means you use a small step for resizing, i.e. reduce size by 5%, you increase the chance of a matching size with the model for detection is found. This also means that the algorithm works slower since it is more thorough. You may increase it to as much as 1.4 for faster detection, with the risk of missing some faces altogether.

		minNeighbors – Parameter specifying how many neighbors each candidate rectangle should have to retain it.

		This parameter will affect the quality of the detected faces. Higher value results in less detections but with higher quality. 3~6 is a good value for it.

		minSize – Minimum possible object size. Objects smaller than that are ignored.

		This parameter determine how small size you want to detect. You decide it! Usually, [30, 30] is a good start for face detection.

		maxSize – Maximum possible object size. Objects bigger than this are ignored.*/
		cascade.detectMultiScale(smallImg, faces,
			1.025, 2, 0
			//|CASCADE_FIND_BIGGEST_OBJECT
			//|CASCADE_DO_ROUGH_SEARCH
			| CASCADE_SCALE_IMAGE,
			Size(30, 120));
		for (vector<Rect>::const_iterator r = faces2.begin(); r != faces2.end(); ++r)
		{
			faces.push_back(Rect(smallImg.cols - r->x - r->width, r->y, r->width, r->height));
		}
	}
	t = (double)getTickCount() - t;
	printf("detection time = %g ms\n", t * 1000 / getTickFrequency()); // détection du visage effectuée a ce niveau la
	for (size_t i = 0; i < faces.size(); i++)		//recherche d'autres élements dans les rectanlges contenant les visages, essayer avec un classifieur visage + yeux? 
	{
		Rect r = faces[i];
		Mat smallImgROI;
		vector<Rect> nestedObjects;	//les objets identifiables dans les rectangles
		Point center;
		Scalar color = colors[i % 8];				//tourner et changer la couleur de l'image augmenterait les chances de détection
		int radius;

		double aspect_ratio = (double)r.width / r.height;
		if (0.75 < aspect_ratio && aspect_ratio < 1.3)
		{
			center.x = cvRound((r.x + r.width*0.5)*scale);
			center.y = cvRound((r.y + r.height*0.5)*scale);
			radius = cvRound((r.width + r.height)*0.25*scale);
			circle(img, center, radius, color, 3, 8, 0);
		}
		else
			rectangle(img, Point(cvRound(r.x*scale), cvRound(r.y*scale)),
				Point(cvRound((r.x + r.width - 1)*scale), cvRound((r.y + r.height - 1)*scale)),
				color, 3, 8, 0);
		if (nestedCascade.empty())
			continue;
		smallImgROI = smallImg(r);
		nestedCascade.detectMultiScale(smallImgROI, nestedObjects,
			1.1, 2, 0
			//|CASCADE_FIND_BIGGEST_OBJECT
			//|CASCADE_DO_ROUGH_SEARCH
			//|CASCADE_DO_CANNY_PRUNING
			| CASCADE_SCALE_IMAGE,
			Size(30, 120 ));
		for (size_t j = 0; j < nestedObjects.size(); j++)
		{
			Rect nr = nestedObjects[j];	// on recupere le rectangle contenant un objet détecté 
			center.x = cvRound((r.x + nr.x + nr.width*0.5)*scale);
			center.y = cvRound((r.y + nr.y + nr.height*0.5)*scale);
			radius = cvRound((nr.width + nr.height)*0.25*scale);
			circle(img, center, radius, color, 3, 8, 0); //on trace un cercle autour de lui et on repete pour chaque obj détecté
		}
	}
	imshow("result", img);			// on affiche le resultat
}


void carDetectAndDraw(Mat& img, CascadeClassifier& cascade,
	double scale)
{
	double t = 0;
	vector<Rect> cars;  // les rectangles qui contiendront les visages
	const static Scalar colors[] =
	{
		Scalar(255,0,0),
		Scalar(255,128,0),
		Scalar(255,255,0),
		Scalar(0,255,0),
		Scalar(0,128,255),
		Scalar(0,255,255),
		Scalar(0,0,255),
		Scalar(255,0,255)
	};
	Mat gray, smallImg;

	cvtColor(img, gray, COLOR_BGR2GRAY);
	double fx = 1 / scale;
	resize(gray, smallImg, Size(), fx, fx, INTER_LINEAR_EXACT);
	equalizeHist(smallImg, smallImg);

	t = (double)getTickCount();
	cascade.detectMultiScale(smallImg, cars,
		1.025, 3, 0
		//|CASCADE_FIND_BIGGEST_OBJECT,
		//|CASCADE_DO_ROUGH_SEARCH,
		//|CASCADE_SCALE_IMAGE,
		| CV_HAAR_DO_CANNY_PRUNING,
		Size(30, 30));

	t = (double)getTickCount() - t;
	printf("detection time = %g ms\n", t * 1000 / getTickFrequency()); // détection du visage effectuée a ce niveau la
	for (size_t i = 0; i < cars.size(); i++)		//recherche d'autres élements dans les rectanlges contenant les visages, essayer avec un classifieur visage + yeux? 
	{
		Rect r = cars[i];
		Mat smallImgROI;
		vector<Rect> nestedObjects;	//les objets identifiables dans les rectangles
		Point center;
		Scalar color = colors[i % 8];	//tourner et changer la couleur de l'image augmenterait les chances de détection
		int radius;

		double aspect_ratio = (double)r.width / r.height;
		if (0.75 < aspect_ratio && aspect_ratio < 1.3)
		{
			center.x = cvRound((r.x + r.width*0.5)*scale);
			center.y = cvRound((r.y + r.height*0.5)*scale);
			radius = cvRound((r.width + r.height)*0.25*scale);
			circle(img, center, radius, color, 3, 8, 0);
		}
		else
			rectangle(img, Point(cvRound(r.x*scale), cvRound(r.y*scale)),
				Point(cvRound((r.x + r.width - 1)*scale), cvRound((r.y + r.height - 1)*scale)),
				color, 3, 8, 0);

		string outString = string("car");
		putText(img, outString, Point(r.x + r.width, r.y + r.height), FONT_HERSHEY_COMPLEX_SMALL, 1.0, Scalar(255, 255, 255));


	}
	imshow("result", img);			// on affiche le resultat
}

void pedestrianDetectAndDraw(Mat& img, CascadeClassifier& cascade, double scale)
{
	double t = 0;
	vector<Rect> pedestrians;  // les rectangles qui contiendront les visages
	const static Scalar colors[] =
	{
		Scalar(255,0,0),
		Scalar(255,128,0),
		Scalar(255,255,0),
		Scalar(0,255,0),
		Scalar(0,128,255),
		Scalar(0,255,255),
		Scalar(0,0,255),
		Scalar(255,0,255)
	};
	Mat gray, smallImg;

	cvtColor(img, gray, COLOR_BGR2GRAY);
	double fx = 1 / scale;
	resize(gray, smallImg, Size(), fx, fx, INTER_LINEAR_EXACT);
	equalizeHist(smallImg, smallImg);

	t = (double)getTickCount();
	cascade.detectMultiScale(smallImg, pedestrians,
		1.025, 3, 0
		//|CASCADE_FIND_BIGGEST_OBJECT,
		//| CASCADE_DO_ROUGH_SEARCH,
		//|CASCADE_SCALE_IMAGE,
		| CV_HAAR_DO_CANNY_PRUNING,
		Size(30, 30));

	t = (double)getTickCount() - t;
	printf("detection time = %g ms\n", t * 1000 / getTickFrequency()); // détection du visage effectuée a ce niveau la
	for (size_t i = 0; i < pedestrians.size(); i++)		//recherche d'autres élements dans les rectanlges contenant les visages, essayer avec un classifieur visage + yeux? 
	{
		Rect r = pedestrians[i];
		Mat smallImgROI;
		vector<Rect> nestedObjects;	//les objets identifiables dans les rectangles
		Point center;
		Scalar color = colors[i % 8];	//tourner et changer la couleur de l'image augmenterait les chances de détection
		int radius;

		double aspect_ratio = (double)r.width / r.height;
		if (0.75 < aspect_ratio && aspect_ratio < 1.3)
		{
			center.x = cvRound((r.x + r.width*0.5)*scale);
			center.y = cvRound((r.y + r.height*0.5)*scale);
			radius = cvRound((r.width + r.height)*0.25*scale);
			circle(img, center, radius, color, 3, 8, 0);
		}
		else
			rectangle(img, Point(cvRound(r.x*scale), cvRound(r.y*scale)),
				Point(cvRound((r.x + r.width - 1)*scale), cvRound((r.y + r.height - 1)*scale)),
				color, 3, 8, 0);

		string outString = string("pedestrian ");
		putText(img, outString, Point(r.x + r.width, r.y + r.height), FONT_HERSHEY_COMPLEX_SMALL, 1.0, Scalar(255, 255, 255));


	}
	imshow("result", img);
}