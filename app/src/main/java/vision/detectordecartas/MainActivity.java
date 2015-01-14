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

    // Drawing constants
    private static final int CONTOUR_THICKNESS = 2;
    private static final Scalar YELLOW = new Scalar(255,255,0,255);
    private static final Scalar BLUE = new Scalar(0,0,255,255);
    private static final Scalar RED = new Scalar(255,0,0,255);
    private static final Scalar GREEN = new Scalar(0,255,0,255);
    // Image processing constants
    private static final int MINIMUM_CARD_SIZE = 17000;
    private static final int MINIMUM_NUMBER_COLOR = 20;
    // Classification constants
    private static final int BASTOS = 0;
    private static final int ESPADAS = 1;
    private static final int COPAS = 2;
    private static final int OROS = 3;
    private static final int NONE = 4;
    private String[] CARD_NAMES = {"BASTOS","ESPADAS","COPAS","OROS","SIN DETERMINAR"};

    // Debugging constants
    private static final int DEBUG = 1;

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
        List<Mat> cCards = new ArrayList<>();
        List<Mat> cCardsOrig = new ArrayList<>();
        int card;
        Mat cardHsv;
        Mat cardElements;
        Mat cardElementsContours;
        Mat cardElementsNoRed;
        Mat cardElementsNoRedContours;
        Mat cardDebug;

        /*
         * ELEMENTS FOR IMAGE PROCESSING
         */
        // Array list of detected contours in the first processing
        List<MatOfPoint> preCardContours = new ArrayList<>();
        List<MatOfPoint> cardContours = new ArrayList<>();
        // Array list of detected contours inside each card
        List<MatOfPoint> cardInsideContours = new ArrayList<>();
        List<MatOfPoint> cardInsideContoursNoRed = new ArrayList<>();
        // Matrix of hierarchy of contours
        Mat mHierarchy = new Mat();

        // HSV color matrix 
        Mat cCardHSV;

        // HSV mask
        Mat greenHsvMask;
        Mat yellowHsvMask;
        Mat blueHsvMask;
        Mat darkRedHsvMask;
        Mat lightRedHsvMask;

        int elementSize;
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
        int total;

        // Drawing elements
        Scalar contourColour = RED;

        // Number of the card
        int nCard = 0;

        // Variables for the treatment of the area of an element of a card
        double areaValue;
        double maxAreaValue;
        double distanceToDouble;
        double distanceToOne;
        boolean twoElementsJoint = false;

        /*
         * ITERATORS
         */
        // Iterator for the lists of MatOfPoint elements
        Iterator<MatOfPoint> eachMop;
        Iterator<Mat> eachCard;
        int i = 0;
        int j = 0;

        /*
         * TEXT ELEMENTS
         */
        double margin;
        double textSize;

        /*
         * IMAGE PROCESSING
         */
        // Binarizing the image to obtain easier the contours (Dilate
        // the image before to reduce noise)
        elementSize = 4;
        element = Imgproc.getStructuringElement(Imgproc.MORPH_RECT,
                new Size(elementSize, elementSize));
        Imgproc.dilate (gi, dilatedGi, element);
        Imgproc.threshold(dilatedGi, processedGi, 150, 255, Imgproc.THRESH_OTSU);

        // Get the contours
        Imgproc.findContours(processedGi, preCardContours, mHierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);

        // Approximate the contours to a polygon
        preCardContours = approximateContours (preCardContours, 10, 4);
        // Draw contours on the mask filling them
        Imgproc.drawContours(maskCardContours, preCardContours, -1, new Scalar(255), -1);
        // Extracting only the content of the contours
        ci.copyTo(processedCi, maskCardContours);

        // Extracting cards with the rectangle around
        eachMop = preCardContours.iterator();
        i = 0;
        while (eachMop.hasNext()){

            contour = eachMop.next();
            
            // Draw card contours with a thickness depending on the size of the card
            contourSize = getSize (contour.toList());
            if (contourSize > MINIMUM_CARD_SIZE){
                thickness = getThickness (contourSize);
                cardContours.add(contour);
                Imgproc.drawContours(processedCi, preCardContours, i, new Scalar (255,255,255,255), thickness);

                rowsCols = getRowsCols(contour);
                    cCard = processedCi.submat(rowsCols[0], rowsCols[1], rowsCols[2], rowsCols[3]);
                cCards.add(cCard);
                cCard = ci.submat(rowsCols[0], rowsCols[1], rowsCols[2], rowsCols[3]);
                cCardsOrig.add (cCard);
            }
            i++;
        }

        // Classify cards
        eachCard = cCards.iterator();
        i = 0;
        while (eachCard.hasNext()){
            cCard = eachCard.next();

            // Card to HSV
            cardHsv = cCard.clone();
            Imgproc.cvtColor(cCard,cardHsv,Imgproc.COLOR_RGBA2RGB);
            Imgproc.cvtColor(cardHsv,cardHsv,Imgproc.COLOR_RGB2HSV);

            // Initializing matrices for processing
            greenHsvMask  = new Mat (cardHsv.size(), CvType.CV_8U, new Scalar(0));
            yellowHsvMask = new Mat (cardHsv.size(), CvType.CV_8U, new Scalar(0));
            blueHsvMask = new Mat (cardHsv.size(), CvType.CV_8U, new Scalar(0));
            darkRedHsvMask = new Mat (cardHsv.size(), CvType.CV_8U, new Scalar(0));
            lightRedHsvMask = new Mat (cardHsv.size(), CvType.CV_8U, new Scalar(0));
            cardElements = new Mat (cCard.size(), CvType.CV_8U, new Scalar(0));
            cardElementsNoRed = new Mat (cCard.size(), CvType.CV_8U, new Scalar(0));
            cardDebug = new Mat (cCard.size(), CvType.CV_8U, new Scalar(0));

            // GREEN
            Core.inRange(cardHsv, new Scalar(35, 40, 50), new Scalar(80, 255, 255), greenHsvMask);
            green = assignValue (Core.countNonZero(greenHsvMask));
            Log.i (TAG, "GREEN: " + Core.countNonZero(greenHsvMask));
            // YELLOW
            Core.inRange(cardHsv, new Scalar(15, 100, 160), new Scalar(35, 255, 255), yellowHsvMask);
            yellow = assignValue (Core.countNonZero(yellowHsvMask));
            Log.i (TAG, "YELLOW: " + Core.countNonZero(yellowHsvMask));
            // BLUE
            Core.inRange(cardHsv, new Scalar(75, 80, 80), new Scalar(135, 255, 255), blueHsvMask);
            blue = assignValue (Core.countNonZero(blueHsvMask));
            Log.i (TAG, "BLUE: " + Core.countNonZero(blueHsvMask));
            // DARK RED
            Core.inRange(cardHsv, new Scalar(150, 120, 120), new Scalar(180, 255, 255), darkRedHsvMask);
            darkRed = assignValue (Core.countNonZero(darkRedHsvMask));
            Log.i (TAG, "DARK RED: " + Core.countNonZero(darkRedHsvMask));
            // LIGHT RED
            Core.inRange(cardHsv, new Scalar(0, 120, 140), new Scalar(12, 255, 255), lightRedHsvMask);
            lightRed = assignValue (Core.countNonZero(lightRedHsvMask));
            Log.i (TAG, "LIGHT RED: " + Core.countNonZero(lightRedHsvMask));

            if (green > 0 && yellow > 0 && darkRed + lightRed > 0){

                // Apply masks for the first decision filter
                Core.inRange(cardHsv, new Scalar(12, 60, 60), new Scalar(150, 255, 255), cardElements);
                darkRedHsvMask.copyTo(cardElements, darkRedHsvMask);
                lightRedHsvMask.copyTo(cardElements, lightRedHsvMask);

                // Dilate the card image to reduce noise
                elementSize = 3;
                element = Imgproc.getStructuringElement(Imgproc.MORPH_DILATE,
                                                        new Size(elementSize, elementSize));
                Imgproc.dilate(cardElements, cardElements,element);


                cardElements.copyTo(cardDebug, cardElements);

                // Get the contours of the card image
                cardElementsContours = cardElements.clone();
                Imgproc.findContours(cardElementsContours, cardInsideContours, mHierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);
                cardInsideContours = approximateContours (cardInsideContours, 1, -1);
                    
                eachMop = cardInsideContours.iterator();
                nCard = 0;
                areaValue = 0;
                while (eachMop.hasNext() && twoElementsJoint == false){
                    contour = eachMop.next();

                    // If the number of points determining the contour
                    // is grater than 20 increment the number of the
                    // card
                    if (contour.total() > 20){
                        nCard++;
                        
                        Log.i (TAG, "The area of the contour is: " + Imgproc.contourArea (contour));
                        if (areaValue == 0){
                            areaValue = Imgproc.contourArea (contour);
                        }

                        // Check if two elements were joint within a contour
                        if (areaValue > Imgproc.contourArea (contour)){
                            maxAreaValue = areaValue;
                            areaValue = Imgproc.contourArea (contour);
                        }
                        else{
                            maxAreaValue = Imgproc.contourArea (contour);
                        }
                        // If the value of the area of the bigger
                        // element is nearer to the value of the
                        // area of the minimum element by two than
                        // to the area of the minimum element, two elements were joint
                        distanceToDouble = Math.abs (maxAreaValue - areaValue*2);
                        distanceToOne = Math.abs (maxAreaValue - areaValue);
                        if (distanceToDouble < distanceToOne){
                            twoElementsJoint = true;
                        }
                    }
                }

                Log.i (TAG, "The number of found contours is:" + nCard);

                if (twoElementsJoint == true){
                    Log.i (TAG, "Found two joint elements");
                    cardElements.setTo(new Scalar (0));
                    // Apply mask for the second decision filter
                    greenHsvMask.copyTo(cardElements, greenHsvMask);

                    // Dilate the card image to reduce noise
                    elementSize = 25;
                    element = Imgproc.getStructuringElement(Imgproc.MORPH_RECT,
                                                            new Size(elementSize, elementSize));
                    Imgproc.dilate(cardElements, cardElements,element);

                    // Get the contours of the card image
                    cardElementsContours.release();
                    cardElementsContours = cardElements.clone();
                    cardInsideContours.clear();
                    Imgproc.findContours(cardElementsContours, cardInsideContours, mHierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);
                    cardInsideContours = approximateContours (cardInsideContours, 1, -1);

                    nCard = cardInsideContours.size()*2 + 1;
                }

                card = BASTOS;
                contourColour = GREEN;
            }
            else if (blue > 0 && yellow > 0 && blue > (darkRed)){
                /* Apply only blue mask */
                blueHsvMask.copyTo(cardElements, blueHsvMask);

                // Dilate the card image to reduce noise
                elementSize = 10;
                element = Imgproc.getStructuringElement(Imgproc.MORPH_DILATE,
                                                        new Size(elementSize, elementSize));
                Imgproc.dilate(cardElements, cardElements,element);

                // Get the contours of the card image
                cardElementsContours = cardElements.clone();
                Imgproc.findContours(cardElementsContours, cardInsideContours, mHierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);
                cardInsideContours = approximateContours (cardInsideContours, 1, -1);

                nCard = cardInsideContours.size();

                card = ESPADAS;
                contourColour = BLUE;
            }
            else{
                /* Apply yellow and red masks */
                yellowHsvMask.copyTo(cardElements, yellowHsvMask);
                darkRedHsvMask.copyTo(cardElements, darkRedHsvMask);
                lightRedHsvMask.copyTo(cardElements, lightRedHsvMask);

                /* Apply all the masks less the corresponding to the red colours */ 
                yellowHsvMask.copyTo(cardElementsNoRed, yellowHsvMask);

                /* 
                 * CHECK DIFFERENCE BETWEEN CONTOURS WITH AND WITHOUT THE RED CHANNEL 
                 */
                // Open the card image to reduce noise
                elementSize = 3;
                element = Imgproc.getStructuringElement(Imgproc.MORPH_OPEN,
                                                        new Size(elementSize, elementSize));
                Imgproc.morphologyEx(cardElementsNoRed, cardElementsNoRed,Imgproc.MORPH_OPEN,element);

                // Get the contours of the card image
                cardElementsNoRedContours = cardElementsNoRed.clone();
                Imgproc.findContours(cardElementsNoRedContours, cardInsideContoursNoRed, mHierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);
                cardInsideContoursNoRed = approximateContours (cardInsideContoursNoRed, 4, -1);

                Log.i (TAG, "The number of found contours without red is:" + cardInsideContoursNoRed.size());

                // Open the card image to reduce noise
                elementSize = 3;
                element = Imgproc.getStructuringElement(Imgproc.MORPH_OPEN,
                                                        new Size(elementSize, elementSize));
                Imgproc.morphologyEx(cardElements, cardElements,Imgproc.MORPH_OPEN,element);

                // Get the contours of the card image
                cardElementsContours = cardElements.clone();
                Imgproc.findContours(cardElementsContours, cardInsideContours, mHierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);
                cardInsideContours = approximateContours (cardInsideContours, 4, -1);

                Log.i (TAG, "The number of found contours with red is:" + cardInsideContours.size());

                if (yellow > 0 && lightRed > 0 && 
                    cardInsideContours.size() == cardInsideContoursNoRed.size()){
                    card = OROS;
                    contourColour = YELLOW;
                    nCard = cardInsideContours.size();
                }
                else if (yellow > 0 && darkRed > 0 && lightRed > 0){
                    cardElements.setTo(new Scalar(0));
                    darkRedHsvMask.copyTo(cardElements, darkRedHsvMask);

                    // Open the card image to reduce noise
                    elementSize = 6;
                    element = Imgproc.getStructuringElement(Imgproc.MORPH_OPEN,
                                                            new Size(elementSize, elementSize));
                    Imgproc.morphologyEx(cardElements, cardElements,Imgproc.MORPH_OPEN,element);

                    // Get the contours of the card image
                    cardElementsContours = cardElements.clone();
                    cardInsideContours.clear();
                    Imgproc.findContours(cardElementsContours, cardInsideContours, mHierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);
                    cardInsideContours = approximateContours (cardInsideContours, 4, -1);
                    
                    Log.i (TAG, "The number of found contours is:" + cardInsideContours.size());

                    nCard = cardInsideContours.size()/2;
                    card = COPAS;
                    contourColour = RED;
                }
                else{
                    card = NONE;
                }
            }
            Log.i (TAG, "\n\n******************CARD: " + CARD_NAMES[card] + " numero: " + nCard + "******************\n\n");

            // If the card was classified, draw the contours with the corresponding colour
            if (card != NONE){
                Imgproc.drawContours(ci, cardContours, i, contourColour, CONTOUR_THICKNESS);
                size = cCardsOrig.get(i).size();
                rows = (int) size.height;
                cols = (int) size.width;

                margin = cols*0.4;
                textSize = cols*0.01;
                Log.i (TAG, "\n\n******************TEXT SIZE: " + textSize + " ******************\n\n");
                Core.putText(cCardsOrig.get(i), Integer.toString(nCard), new Point(margin, margin), 3, textSize, contourColour, CONTOUR_THICKNESS);
                Imgproc.drawContours(cCard, cardInsideContours, -1, RED, CONTOUR_THICKNESS);
            }

            if (DEBUG == 1){
                rows = (int) allSize.height;
                cols = (int) allSize.width;

                // Set at the corner of the image in gray scale the zoomed card detected
                zoomCorner = gi.submat(0, rows / 2 - rows / 10, 0, cols / 2 - cols / 10);
                Imgproc.resize(cardDebug, zoomCorner, zoomCorner.size());

                // Set at the corner of the image with color the zoomed card detected
                zoomCorner = ci.submat(0, rows / 2 - rows / 10, 0, cols / 2 - cols / 10);
                Imgproc.resize(cCard, zoomCorner, zoomCorner.size());

                // Set at the corner of the image with color the zoomed card detected
                zoomCorner = ciHSV.submat(0, rows / 2 - rows / 10, 0, cols / 2 - cols / 10);
                Imgproc.resize(cardHsv, zoomCorner, zoomCorner.size());
                // Log.i (TAG, "Found contours on the card: " + cardInsideContours.size());
            }

            i++;
        }

        // Free memory
        /*mHierarchy.release();
        ci.release();
        gray_image.release();*/
        return ci;
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

    // protected int[] getContours (){
        
        
        
    // }
}

