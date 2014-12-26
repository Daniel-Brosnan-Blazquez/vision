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
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

public class MainActivity  extends Activity implements CvCameraViewListener2 {
    private static final String TAG = "VISION::Activity";

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

        Mat mHierarchy = new Mat();
        Mat color_image = inputFrame.rgba();
        Mat gray_image = inputFrame.gray();
        Scalar CONTOUR_COLOR = new Scalar(255,0,0,255);
        int CONTOUR_THICKNESS = 2;

        List<MatOfPoint> contours = new ArrayList<MatOfPoint>();

        // Binarizing the image to easier obtain the contours
        Imgproc.threshold(gray_image, gray_image, 150, 255, Imgproc.THRESH_OTSU);

        // Get the contours
        Imgproc.findContours(gray_image, contours, mHierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);

        // Approximate the contours to a rectangle
        List<MatOfPoint> cardContours = new ArrayList<MatOfPoint>();
        Iterator<MatOfPoint> each = contours.iterator();
        while (each.hasNext()){
            MatOfPoint contour = each.next();
            MatOfPoint2f toBeApproxContour = new MatOfPoint2f(contour.toArray());
            MatOfPoint2f approxContour = new MatOfPoint2f();
            Imgproc.approxPolyDP (toBeApproxContour, approxContour, 10, true);
            if (approxContour.total() == 4) {
                for (Point point : approxContour.toArray())
                    Log.i(TAG, "Contour: " + point);
                MatOfPoint aContour = new MatOfPoint(approxContour.toArray());
                cardContours.add(aContour);
            }
        }

        // Draw contours on the image
        Imgproc.drawContours(color_image, cardContours, -1, CONTOUR_COLOR, CONTOUR_THICKNESS);

        return color_image;
    }

}

