package vision.detectordecartas;

import android.app.Activity;
import android.os.Bundle;
import android.util.Log;
import android.view.SurfaceView;
import android.view.WindowManager;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewFrame;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewListener2;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;

public class MainActivity  extends Activity implements CvCameraViewListener2 {
    private static final String TAG = "VISION::Activity";
    private static final Scalar CONTOUR_COLOR = new Scalar(255,0,0,255);
    private static final int CONTOUR_THICKNESS = 2;
    private static final Scalar CARD_CONTOUR_COLOR = new Scalar(0,0,255,255);
    private static final int MINIMUM_CARD_SIZE = 17000;
    private static final int MINIMUM_NUMBER_COLOR = 50;
    private static final int NONE = 4;
    private static final int BASTOS = 0;
    private static final int ESPADAS = 1;
    private static final int COPAS = 2;
    private static final int OROS = 3;
    private String[] cardNames = new String[20];

    private CameraBridgeViewBase mOpenCvCameraView;

    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        // Method to enable view after the load of the open cv library
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS:
                {
                    Log.i(TAG, "OpenCV loaded successfully");
                    mOpenCvCameraView.enableView();
                } break;
                default:
                {
                    super.onManagerConnected(status);
                } break;
            }
        }
    };

    @Override
    protected void onCreate(Bundle savedInstanceState) {

        Log.i(TAG, "called onCreate");
        super.onCreate(savedInstanceState);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        setContentView(R.layout.activity_main);

        mOpenCvCameraView = (CameraBridgeViewBase) findViewById(R.id.java_view);
        mOpenCvCameraView.setVisibility(SurfaceView.VISIBLE);
        mOpenCvCameraView.setCvCameraViewListener(this);
    }

    @Override
    public void onPause()
    {
        super.onPause();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    @Override
    public void onResume()
    {
        super.onResume();
        OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_2_4_3, this, mLoaderCallback);

        Log.i(TAG, "called onResume");

    }

    public void onDestroy() {
        super.onDestroy();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    public void onCameraViewStarted(int width, int height) {
    }

    public void onCameraViewStopped() {
    }

    public Mat onCameraFrame(CvCameraViewFrame inputFrame) {

        Log.i(TAG, "called onCameraFrame");

        /*
         * COLOR IMAGE
         */
        // Original image in RGB mode
        Mat ci = inputFrame.rgba();
        Size allSize = ci.size();
        // Matrix to store the processed image in RGB mode
        Mat processedCi = new Mat (allSize, CvType.CV_8UC4, new Scalar(250,250,250,250));
        Mat maskCardContours = new Mat (ci.size(), CvType.CV_8U, new Scalar(0));
        Mat ciHSV = ci.clone();
        Imgproc.cvtColor(ci,ciHSV,Imgproc.COLOR_RGBA2RGB);
        Imgproc.cvtColor(ciHSV,ciHSV,Imgproc.COLOR_RGB2HSV);

        /*
         * GRAY IMAGE
         */
        // Original image in GRAY mode
        Mat gi = inputFrame.gray();
        // Matrix to store the processed image in gray mode
        Mat processedGi = new Mat (gi.size(), CvType.CV_8U, new Scalar(0));
        Mat dilatedGi = new Mat (gi.size(), CvType.CV_8U, new Scalar(0));

        /*
         * CARDS
         */
        Mat cCard;
        Mat gCard;
        Mat gCardFC;
        List<Mat> cCards = new ArrayList<Mat>();
        int card;
        cardNames[BASTOS] = "BASTOS";
        cardNames[ESPADAS] = "ESPADAS";
        cardNames[COPAS] = "COPAS";
        cardNames[OROS] = "OROS";
        cardNames[NONE] = "SIN DETERMINAR";

        /*
         * ELEMENTS FOR IMAGE PROCESSING
         */
        // Array list of detected contours in the first processing
        List<MatOfPoint> cardContours = new ArrayList<MatOfPoint>();
        // Array list of detected contours inside each card
        List<MatOfPoint> cardInsideContours = new ArrayList<MatOfPoint>();
        // Matrix of hierarchy of contours
        Mat mHierarchy = new Mat();
        // Matrix to be return
        int dilationSize;
        Mat element;
        MatOfPoint contour;
        int[] rowsCols;
        int rows;
        int cols;
        Size size;
        Mat zoomCorner;
        int thickness;
        double contourSize;
        // counters of colors
        int green;
        int blue;
        int lightRed;
        int darkRed;
        int yellow;

        /*
         * ITERATORS
         */
        // Iterator for the lists of MatOfPoint elements
        Iterator<MatOfPoint> eachMop;
        int i = 0;

        /*
         * IMAGE PROCESSING
         */
        // Binarizing the image to easier obtain the contours (Dilate
        // the image before to reduce noise)
        dilationSize = 4;
        element = Imgproc.getStructuringElement(Imgproc.MORPH_RECT,
                new Size(dilationSize, dilationSize));
        Imgproc.dilate (gi, dilatedGi, element);
        Imgproc.threshold(dilatedGi, processedGi, 150, 255, Imgproc.THRESH_OTSU);

        // Get the contours
        Imgproc.findContours(processedGi, cardContours, mHierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);

        // Approximate the contours to a polygon
        cardContours = approximateContours (cardContours, 10, 4);
        // Draw contours on the mask filling them
        Imgproc.drawContours(maskCardContours, cardContours, -1, new Scalar(255), -1);
        // Extracting only the content of the contours
        ci.copyTo(processedCi, maskCardContours);

        // Extracting cards with the rectangle around
        eachMop = cardContours.iterator();
        i = 0;
        while (eachMop.hasNext()){

            contour = eachMop.next();
            
            // Draw card contours with a thickness depending on the size of the card
            contourSize = getSize (contour.toList());
            if (contourSize > MINIMUM_CARD_SIZE){
                thickness = getThickness (contourSize);
                Imgproc.drawContours(processedCi, cardContours, i, new Scalar (255,255,255,255), thickness);

                rowsCols = getRowsCols(contour);
                cCard = processedCi.submat(rowsCols[0], rowsCols[1], rowsCols[2], rowsCols[3]);
                cCards.add (cCard);
            }
            i++;
        }

        // If there were found contours, process them
        if (cCards.size() > 0) {
            cCard = cCards.get(0);
            size = cCard.size();
            rows = (int) allSize.height;
            cols = (int) allSize.width;

            /*// Dilate the card image to reduce noise
            dilationSize = 5;
            element = Imgproc.getStructuringElement(Imgproc.MORPH_DILATE,
                                                    new Size(dilationSize, dilationSize));
            Imgproc.dilate (cCard, cCard, element);*/

            // // Erosion
            // dilationSize = 5;
            // element = Imgproc.getStructuringElement(Imgproc.MORPH_ERODE,
            //                                         new Size(dilationSize, dilationSize));
            // Imgproc.erode (cCard, cCard, element);

            // Card to HSV
            Mat cCardHSV = cCard.clone();
            Imgproc.cvtColor(cCard,cCardHSV,Imgproc.COLOR_RGBA2RGB);
            Imgproc.cvtColor(cCardHSV,cCardHSV,Imgproc.COLOR_RGB2HSV);

            Mat cCardHSV2 = cCardHSV.clone();
            // GREEN
            Core.inRange(cCardHSV, new Scalar(35, 10, 20), new Scalar(80, 255, 255), cCardHSV2);
            green = assignValue (Core.countNonZero(cCardHSV2));
            Log.i (TAG, "GREEN: " + green);
            // YELLOW
            Core.inRange(cCardHSV, new Scalar(15, 100, 100), new Scalar(35, 255, 255), cCardHSV2);
            yellow = assignValue (Core.countNonZero(cCardHSV2));
            Log.i (TAG, "YELLOW: " + yellow);
            // DARK RED
            Core.inRange(cCardHSV, new Scalar(170, 100, 100), new Scalar(180, 255, 255), cCardHSV2);
            darkRed = assignValue (Core.countNonZero(cCardHSV2));
            Log.i (TAG, "DARK RED: " + darkRed);
            // LIGHT RED
            Core.inRange(cCardHSV, new Scalar(0, 40, 60), new Scalar(15, 255, 255), cCardHSV2);
            lightRed = assignValue (Core.countNonZero(cCardHSV2));
            Log.i (TAG, "LIGHT RED: " + lightRed);
            // BLUE
            Core.inRange(cCardHSV, new Scalar(75, 20, 80), new Scalar(135, 255, 255), cCardHSV2);
            blue = assignValue (Core.countNonZero(cCardHSV2));
            Log.i (TAG, "BLUE: " + blue);

            // GREEN,RED y YELLOW
            //Core.inRange(cCardHSV, new Scalar(0, 5, 5), new Scalar(60, 255, 255), cCardHSV2);
            //Log.i (TAG, "GREEN,RED y YELLOW: " + Core.countNonZero(cCardHSV2));

            try {
                Thread.sleep(1000, 0);
            }catch(InterruptedException e) {
                e.printStackTrace();
            }

            if (green > 0 && yellow > 0 && darkRed > 0 && lightRed > 0)
                card = BASTOS;
            else if (blue > 0 && yellow > 0 && blue > (darkRed + lightRed))
                card = ESPADAS;
            else if (yellow > 0 && darkRed > 0 && lightRed > 0)
                card = COPAS;
            else if (yellow > 0 && lightRed > 0)
                card = OROS;
            else
                card = NONE;

            Log.i (TAG, "******************\n\nCARD: " + cardNames[card] + "\n\n******************");


            /*Imgproc.cvtColor(cCardHSV,cCardHSV, Imgproc.COLOR_GRAY2RGB, 4);
            Imgproc.cvtColor(cCardHSV,cCardHSV,Imgproc.COLOR_RGB2HSV);*/

            // // Dilate the card image to reduce noise
            // dilationSize = 5;
            // element = Imgproc.getStructuringElement(Imgproc.MORPH_ERODE,
            //                                         new Size(dilationSize, dilationSize));
            // Imgproc.erode (cCardHSV, cCardHSV, element);


            // Get the card image in gray scale and binarize it
/*            gCard=new Mat (cCard.size(), CvType.CV_8U);
            Imgproc.cvtColor (cCardHSV, gCard, Imgproc.COLOR_RGBA2GRAY);*/

            //            Imgproc.Canny (gCard, gCard, 10,100);

            //Imgproc.threshold(gCard, gCard, 150, 255, Imgproc.THRESH_OTSU);

            // // // Dilate the card in gray scale to reduce noise
            // // dilationSize = 4;
            // // element = Imgproc.getStructuringElement(Imgproc.MORPH_RECT,
            // //                                         new Size(dilationSize, dilationSize));
            // // Imgproc.dilate (gCard, gCard, element);

            // // dilationSize = 10;
            // // element = Imgproc.getStructuringElement(Imgproc.MORPH_RECT,
            // //                                         new Size(dilationSize, dilationSize));
            // // Imgproc.erode (gCard, gCard, element);

            // // //            Imgproc.threshold(gCard, gCard, 150, 255, Imgproc.THRESH_BINARY);
            
            // Get the contours of the card image
            /*gCardFC = gCard.clone();
            Imgproc.findContours(gCardFC, cardInsideContours, mHierarchy, Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_SIMPLE);
            cardInsideContours = approximateContours (cardInsideContours, 4, -1);

            // Draw contours
            Imgproc.drawContours(cCard, cardInsideContours, -1, CARD_CONTOUR_COLOR, CONTOUR_THICKNESS);
          */
            // Set at the corner of the image in gray scale the zoomed card detected
            zoomCorner = gi.submat(0, rows / 2 - rows / 10, 0, cols / 2 - cols / 10);
            Imgproc.resize(cCardHSV2, zoomCorner, zoomCorner.size());

            // Set at the corner of the image with color the zoomed card detected
            zoomCorner = ci.submat(0, rows / 2 - rows / 10, 0, cols / 2 - cols / 10);
            Imgproc.resize(cCard, zoomCorner, zoomCorner.size());

            // Set at the corner of the image with color the zoomed card detected
            zoomCorner = ciHSV.submat(0, rows / 2 - rows / 10, 0, cols / 2 - cols / 10);
            Imgproc.resize(cCardHSV, zoomCorner, zoomCorner.size());
            // Log.i (TAG, "Found contours on the card: " + cardInsideContours.size());
        }

        // Free memory
        /*mHierarchy.release();
        ci.release();
        gray_image.release();*/
        return gi;
    }

    protected int[] getRowsCols (MatOfPoint contour) {
        int minRow = -1;
        int maxRow = -1;
        int minCol = -1;
        int maxCol = -1;

        for (Point point : contour.toArray()){
            int row = (int) point.y;
            int col = (int) point.x;
            if (row > maxRow || maxRow == -1)
                maxRow = row;
            if (row < minRow || minRow == -1)
                minRow = row;
            if (col > maxCol || maxCol == -1)
                maxCol = col;
            if (col < minCol || minCol == -1)
                minCol = col;
        }
        int[] rowsCols = {minRow, maxRow, minCol, maxCol};
        return rowsCols;
    }

    protected List<MatOfPoint> approximateContours 
        (List<MatOfPoint> contours, int epsilon, int nPoints){
        MatOfPoint contour;
        MatOfPoint2f toBeApproxContour;
        MatOfPoint2f approxContour;
        MatOfPoint aContour;
        List<MatOfPoint> approxContours = new ArrayList<MatOfPoint>();

        Iterator<MatOfPoint> each = contours.iterator();
        while (each.hasNext()){
            contour = each.next();
            toBeApproxContour = new MatOfPoint2f(contour.toArray());
            approxContour = new MatOfPoint2f();
            Imgproc.approxPolyDP (toBeApproxContour, approxContour, epsilon, true);
            aContour = new MatOfPoint(approxContour.toArray());
            // If nPoints is negative return all the contours
            if (approxContour.total() == nPoints) {
                approxContours.add(aContour);
            }
            else if (nPoints < 0){
                approxContours.add (aContour);
            }
        }
        return approxContours;
    }

    protected double getSize (List<Point> cardContour){

        Iterator<Point> each = cardContour.iterator();
        double[] distances = {0,0,0,0};
        double distance;
        double xC;
        double yC;
        double xN;
        double yN;
        double size;
        Point firstPoint = each.next();
        Point currentPoint = firstPoint;
        Point nextPoint;
        int i = 0;
        while (each.hasNext()) {
            nextPoint = each.next();
            xC = currentPoint.x;
            yC = currentPoint.y;
            xN = nextPoint.x;
            yN = nextPoint.y;
            distance = Math.sqrt(Math.pow(xC-xN,2) + Math.pow(yC-yN,2));
            distances[i] = distance;
            i++;
            currentPoint = nextPoint;
        }

        // Insert the distance between the last and the first points
        xC = currentPoint.x;
        yC = currentPoint.y;
        xN = firstPoint.x;
        yN = firstPoint.y;
        distance = Math.sqrt(Math.pow(xC-xN,2) + Math.pow(yC-yN,2));
        distances[i] = distance;

        Arrays.sort(distances);

        // The size of the rectangle is the area given by the larger side and the largest of the shorter sides
        size = distances[1]*distances[3];

        // Log.i(TAG,"SIZE: " + size);

        return size;

    }

    protected int getThickness (double size){
        
        /*
         * 9000  -> 20
         *   X   -> Y
         * 17000 -> 70
         * Y = 20 + (X - 9000)*((70 - 20)/(17000 - 9000))
         */

        int thickness = (int) (25 + (size - 9000.0)*((80.0 - 25.0)/(170000.0 - 9000.0)));

        return thickness;
    }

    protected int assignValue (int n){
        
        int ret = n;
        if (n < MINIMUM_NUMBER_COLOR){
            ret = 0;
        }
        return ret;
    }
}

