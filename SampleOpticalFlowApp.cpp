#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
//#include <opencv2/videoio.hpp>
#include <opencv2/video.hpp>
using namespace cv;
using namespace std;

#define DEBUG_CODE 0
#define  CV_DESCALE(x,n)     (((x) + (1 << ((n)-1))) >> (n))

typedef float acctype;
typedef float itemtype;

void opticalFlowLK_singlePyrLvl(const Mat& prevImg, const Mat& prevDeriv, const Mat& nextImg,
                      const Point2f* prevPts, Point2f* nextPts,
                      uchar* status, float* err,
                      Size winSize, TermCriteria criteria,
                      int level, int maxLevel, int flags, float minEigThreshold, int npoints ) {


    printf("kumuda Priya: Entered LKTracker operator\n");

    Point2f halfWin((winSize.width-1)*0.5f, (winSize.height-1)*0.5f);
    const Mat& I = *(&prevImg);
    const Mat& J = *(&nextImg);
    const Mat& derivI = *(&prevDeriv);

    int j, cn = I.channels(), cn2 = cn*2;
    cv::AutoBuffer<short> _buf(winSize.area()*(cn + cn2));
    int derivDepth = 3;

    Mat IWinBuf(winSize, CV_MAKETYPE(derivDepth, cn), _buf.data());
    Mat derivIWinBuf(winSize, CV_MAKETYPE(derivDepth, cn2), _buf.data() + winSize.area()*cn);

    for( int ptidx = 0; ptidx < npoints; ptidx++ )
    {
#if DEBUG_CODE
        if(ptidx != 16)
            continue;
#endif
        
        Point2f prevPt = prevPts[ptidx]*(float)(1./(1 << level));
#if DEBUG_CODE
        printf("ptidx : %d I_Pt (%f,%f)\n",ptidx, prevPts[ptidx].x, prevPts[ptidx].y);
        printf("Level : %d I_Pt_level (%f,%f)\n", level,  prevPt.x, prevPt.y);
#endif

        Point2f nextPt;
        if( level == maxLevel )
        {
            if( flags & OPTFLOW_USE_INITIAL_FLOW )
                nextPt = nextPts[ptidx]*(float)(1./(1 << level));
            else
                nextPt = prevPt;
        }
        else
            nextPt = nextPts[ptidx]*2.f;
        nextPts[ptidx] = nextPt;

#if DEBUG_CODE
        printf("J_Pt_level (%f,%f)\n", nextPt.x, nextPt.y);
#endif

        Point2i iprevPt, inextPt;
        prevPt -= halfWin;
        iprevPt.x = cvFloor(prevPt.x);
        iprevPt.y = cvFloor(prevPt.y);

        if( iprevPt.x < -winSize.width || iprevPt.x >= derivI.cols ||
            iprevPt.y < -winSize.height || iprevPt.y >= derivI.rows )
        {
            if( level == 0 )
            {
                if( status )
                    status[ptidx] = false;
                if( err )
                    err[ptidx] = 0;
            }
            continue;
        }

        float a = prevPt.x - iprevPt.x;
        float b = prevPt.y - iprevPt.y;
        const int W_BITS = 14, W_BITS1 = 14;
        const float FLT_SCALE = 1.f/(1 << 20);
        int iw00 = cvRound((1.f - a)*(1.f - b)*(1 << W_BITS));
        int iw01 = cvRound(a*(1.f - b)*(1 << W_BITS));
        int iw10 = cvRound((1.f - a)*b*(1 << W_BITS));
        int iw11 = (1 << W_BITS) - iw00 - iw01 - iw10;

        int dstep = (int)(derivI.step/derivI.elemSize1());
        int stepI = (int)(I.step/I.elemSize1());
        int stepJ = (int)(J.step/J.elemSize1());
        acctype iA11 = 0, iA12 = 0, iA22 = 0;
        float A11, A12, A22;

#if DEBUG_CODE
        printf("Left top corner I_Pt (%d,%d) + (%f,%f)\n", iprevPt.x, iprevPt.y, a, b);
#endif

        // extract the patch from the first image, compute covariation matrix of derivatives
        int x, y;
        for( y = 0; y < winSize.height; y++ )
        {
            const uchar* src = I.ptr() + (y + iprevPt.y)*stepI + iprevPt.x*cn;
            const short* dsrc = derivI.ptr<short>() + (y + iprevPt.y)*dstep + iprevPt.x*cn2;

            short* Iptr = IWinBuf.ptr<short>(y);
            short* dIptr = derivIWinBuf.ptr<short>(y);

            x = 0;
            for( ; x < winSize.width*cn; x++, dsrc += 2, dIptr += 2 )
            {
                int ival = CV_DESCALE(src[x]*iw00 + src[x+cn]*iw01 +
                                      src[x+stepI]*iw10 + src[x+stepI+cn]*iw11, W_BITS1-5);
                int ixval = CV_DESCALE(dsrc[0]*iw00 + dsrc[cn2]*iw01 +
                                       dsrc[dstep]*iw10 + dsrc[dstep+cn2]*iw11, W_BITS1);
                int iyval = CV_DESCALE(dsrc[1]*iw00 + dsrc[cn2+1]*iw01 + dsrc[dstep+1]*iw10 +
                                       dsrc[dstep+cn2+1]*iw11, W_BITS1);

                Iptr[x] = (short)ival;
                dIptr[0] = (short)ixval;
                dIptr[1] = (short)iyval;

#if DEBUG_CODE

                int c = x+iprevPt.x, d = y + iprevPt.y; 
                printf("Interpolated I[(%d,%d)] : %d, dx: %d  dy : %d\n", iprevPt.x + x, iprevPt.y + y, Iptr[x], dIptr[0], dIptr[1]);
                printf("Compute dx : \n");
                printf("dx[%d,%d] : %d dx[%d,%d] : %d dx[%d,%d] : %d dx[%d,%d] : %d\n",c,d,dsrc[0],c+1,d,dsrc[cn2],c,d+1,dsrc[dstep],c+1,d+1,dsrc[dstep+cn2]);
                printf("Compute dy : \n");
                printf("dy[%d,%d] : %d dy[%d,%d] : %d dy[%d,%d] : %d dy[%d,%d] : %d\n",c,d,dsrc[1],c+1,d,dsrc[cn2+1],c,d+1,dsrc[dstep+1],c+1,d+1,dsrc[dstep+cn2+1]);

#endif

                iA11 += (itemtype)(ixval*ixval);
                iA12 += (itemtype)(ixval*iyval);
                iA22 += (itemtype)(iyval*iyval);
            }
        }

        A11 = iA11*FLT_SCALE;
        A12 = iA12*FLT_SCALE;
        A22 = iA22*FLT_SCALE;

#if DEBUG_CODE
        printf("Gxx : %f, Gxy : %f, Gyy:%f\n", A11, A12, A22);
#endif

        float D = A11*A22 - A12*A12;
        float minEig = (A22 + A11 - std::sqrt((A11-A22)*(A11-A22) +
                        4.f*A12*A12))/(2*winSize.width*winSize.height);

        if( err && (flags & OPTFLOW_LK_GET_MIN_EIGENVALS) != 0 )
            err[ptidx] = (float)minEig;

        if( minEig < minEigThreshold || D < FLT_EPSILON )
        {
            if( level == 0 && status )
                status[ptidx] = false;
            continue;
        }

        D = 1.f/D;

        nextPt -= halfWin;
        Point2f prevDelta;

        for( j = 0; j < criteria.maxCount; j++ )
        {
            inextPt.x = cvFloor(nextPt.x);
            inextPt.y = cvFloor(nextPt.y);

            if( inextPt.x < -winSize.width || inextPt.x >= J.cols ||
               inextPt.y < -winSize.height || inextPt.y >= J.rows )
            {
                if( level == 0 && status )
                    status[ptidx] = false;
                break;
            }

            a = nextPt.x - inextPt.x;
            b = nextPt.y - inextPt.y;
            iw00 = cvRound((1.f - a)*(1.f - b)*(1 << W_BITS));
            iw01 = cvRound(a*(1.f - b)*(1 << W_BITS));
            iw10 = cvRound((1.f - a)*b*(1 << W_BITS));
            iw11 = (1 << W_BITS) - iw00 - iw01 - iw10;
            acctype ib1 = 0, ib2 = 0;
            float b1, b2;

            for( y = 0; y < winSize.height; y++ )
            {
                const uchar* Jptr = J.ptr() + (y + inextPt.y)*stepJ + inextPt.x*cn;
                const short* Iptr = IWinBuf.ptr<short>(y);
                const short* dIptr = derivIWinBuf.ptr<short>(y);

                x = 0;
                for( ; x < winSize.width*cn; x++, dIptr += 2 )
                {
                    int diff = CV_DESCALE(Jptr[x]*iw00 + Jptr[x+cn]*iw01 +
                                          Jptr[x+stepJ]*iw10 + Jptr[x+stepJ+cn]*iw11,
                                          W_BITS1-5) - Iptr[x];
                    ib1 += (itemtype)(diff*dIptr[0]);
                    ib2 += (itemtype)(diff*dIptr[1]);
                }
            }

            b1 = ib1*FLT_SCALE;
            b2 = ib2*FLT_SCALE;

            Point2f delta( (float)((A12*b2 - A22*b1) * D),
                          (float)((A12*b1 - A11*b2) * D));
            //delta = -delta;

            nextPt += delta;
            nextPts[ptidx] = nextPt + halfWin;

            if( delta.ddot(delta) <= criteria.epsilon )
                break;

            if( j > 0 && std::abs(delta.x + prevDelta.x) < 0.01 &&
               std::abs(delta.y + prevDelta.y) < 0.01 )
            {
                nextPts[ptidx] -= delta*0.5f;
                break;
            }
            prevDelta = delta;
        }

        CV_Assert(status != NULL);
        if( status[ptidx] && err && level == 0 && (flags & OPTFLOW_LK_GET_MIN_EIGENVALS) == 0 )
        {
            Point2f nextPoint = nextPts[ptidx] - halfWin;
            Point inextPoint;

            inextPoint.x = cvFloor(nextPoint.x);
            inextPoint.y = cvFloor(nextPoint.y);

            if( inextPoint.x < -winSize.width || inextPoint.x >= J.cols ||
                inextPoint.y < -winSize.height || inextPoint.y >= J.rows )
            {
                if( status )
                    status[ptidx] = false;
                continue;
            }

            float aa = nextPoint.x - inextPoint.x;
            float bb = nextPoint.y - inextPoint.y;
            iw00 = cvRound((1.f - aa)*(1.f - bb)*(1 << W_BITS));
            iw01 = cvRound(aa*(1.f - bb)*(1 << W_BITS));
            iw10 = cvRound((1.f - aa)*bb*(1 << W_BITS));
            iw11 = (1 << W_BITS) - iw00 - iw01 - iw10;
            float errval = 0.f;

            for( y = 0; y < winSize.height; y++ )
            {
                const uchar* Jptr = J.ptr() + (y + inextPoint.y)*stepJ + inextPoint.x*cn;
                const short* Iptr = IWinBuf.ptr<short>(y);

                for( x = 0; x < winSize.width*cn; x++ )
                {
                    int diff = CV_DESCALE(Jptr[x]*iw00 + Jptr[x+cn]*iw01 +
                                          Jptr[x+stepJ]*iw10 + Jptr[x+stepJ+cn]*iw11,
                                          W_BITS1-5) - Iptr[x];
                    errval += std::abs((float)diff);
                }
            }
            err[ptidx] = errval * 1.f/(32*winSize.width*cn*winSize.height);
        }
    }
}

void ownOpticalFlowLK(InputArray _prevImg, InputArray _nextImg,
                           InputArray _prevPts, InputOutputArray _nextPts,
                           OutputArray _status, OutputArray _err,
                           Size winSize = Size(21,21), int maxLevel = 3,
                                        TermCriteria criteria = TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 30, 0.01),
                                        int flags = 0, double minEigThreshold = 1e-4 ) {
    

    //Need to Add Asset Condtions on parameters
    int i,npoints;
    Mat prevPtsMat = _prevPts.getMat();
    npoints = prevPtsMat.checkVector(2, CV_32F, true);

    if( !(flags & OPTFLOW_USE_INITIAL_FLOW) ) {
        _nextPts.create(prevPtsMat.size(), prevPtsMat.type(), -1, true);
        printf("%s : Copy the previous points to _nextPts", __func__);
    }

    Mat nextPtsMat = _nextPts.getMat();
    const Point2f* prevPts = prevPtsMat.ptr<Point2f>();
    Point2f* nextPts = nextPtsMat.ptr<Point2f>();

    _status.create((int)npoints, 1, CV_8U, -1, true);
    Mat statusMat = _status.getMat(), errMat;
    CV_Assert( statusMat.isContinuous() );
    uchar* status = statusMat.ptr();
    float* err = 0;

    for( i = 0; i < npoints; i++ )
        status[i] = true;

    if( _err.needed() )
    {
        _err.create((int)npoints, 1, CV_32F, -1, true);
        errMat = _err.getMat();
        CV_Assert( errMat.isContinuous() );
        err = errMat.ptr<float>();
    }

    int lvlStep1 = 2, lvlStep2 = 2;
    vector<Mat> prevPyr, nextPyr;
    maxLevel = buildOpticalFlowPyramid(_prevImg, prevPyr, winSize, maxLevel, true);
    maxLevel = buildOpticalFlowPyramid(_nextImg, nextPyr, winSize, maxLevel, true);

    if( (criteria.type & TermCriteria::COUNT) == 0 )
        criteria.maxCount = 30;
    else
        criteria.maxCount = std::min(std::max(criteria.maxCount, 0), 100);
    if( (criteria.type & TermCriteria::EPS) == 0 )
        criteria.epsilon = 0.01;
    else
        criteria.epsilon = std::min(std::max(criteria.epsilon, 0.), 10.);
    criteria.epsilon *= criteria.epsilon;

    for(int level = maxLevel; level >= 0; level-- ) {
        Mat derivI;
        derivI = prevPyr[level * lvlStep1 + 1];

        opticalFlowLK_singlePyrLvl(prevPyr[level * lvlStep1], derivI,
                                                          nextPyr[level * lvlStep2], prevPts, nextPts,
                                                          status, err,
                                                          winSize, criteria, level, maxLevel,
                                                          flags, (float)minEigThreshold, npoints);

    }

}



//Build Optical Flow Pyramid


//Build Sharr Derivative

void printMat(Mat &A) {
    cout << "rows : " << A.rows << " cols : " << A.cols << "\n" << endl;
    for(int i=0; i<A.rows; i++) {
        unsigned char *I = A.ptr<unsigned char>(i);
        for(int j=0; j<A.cols; j++) {
            printf("%d ", I[j]);
        }
        printf("\n");
    }
    cout << endl;
}


int main(int argc, char **argv)
{
    
    // Create some random colors
    vector<Scalar> colors;
    RNG rng;
    for(int i = 0; i < 100; i++)
    {
        int r = rng.uniform(0, 256);
        int g = rng.uniform(0, 256);
        int b = rng.uniform(0, 256);
        colors.push_back(Scalar(r,g,b));
    }
    Mat old_frame, old_gray;
    vector<Point2f> p0, p1;
    
    old_frame = imread("frame10_army.png", IMREAD_COLOR);   // Read the file
    cvtColor(old_frame, old_gray, COLOR_BGR2GRAY);
    goodFeaturesToTrack(old_gray, p0, 100, 0.3, 7, Mat(), 7, false, 0.04);
    // Create a mask image for drawing purposes
    Mat mask = Mat::zeros(old_frame.size(), old_frame.type());

    Mat frame, frame_gray;
    frame = imread("frame11_army.png", IMREAD_COLOR);//capture >> frame;
    if (frame.empty()) {
        printf("Empty frame\n");
        return -1;
    }
    cvtColor(frame, frame_gray, COLOR_BGR2GRAY);

    
    // calculate optical flow
    vector<uchar> status;
    vector<float> err;
    TermCriteria criteria = TermCriteria((TermCriteria::COUNT) + (TermCriteria::EPS), 10, 0.03);

#if 0
    Mat dummy_mat;
    dummy_mat.create(20,20,CV_8UC1);


    for(int i=0; i<dummy_mat.rows; i++) {
        unsigned char *I = dummy_mat.ptr<unsigned char>(i);
        for(int j=0; j<dummy_mat.cols; j++) {
            if(j == 0 || i == 0 )
                I[j] = 64;
            else
                I[j] = 0;

            //I[j] = j+1+i;
        }
    }
    
    printMat(dummy_mat);
    cout << endl;

    vector<Mat> pyr;
    buildOpticalFlowPyramid(dummy_mat, pyr, Size(3,3), 2, true);
    printMat(pyr[2]);
#endif

#if 0
    calcOpticalFlowPyrLK(old_gray, frame_gray, p0, p1, status, err, Size(15,15), 2, criteria);
 
    FILE *fp_opencv;
    fp_opencv = fopen("opencv_optical_flow_output.txt", "w+");
    if(fp_opencv == NULL) {
        cout << "Could not open OpenCV Optical Flow file" << endl;
        return -1;
    }


    vector<Point2f> good_new;
    for(uint i = 0; i < p0.size(); i++)
    {
        fprintf(fp_opencv, "Feature Point%d : (%f,%f) => ", i+1, p0[i].x, p0[i].y);
        // Select good points
        if(status[i] == 1) {
            fprintf(fp_opencv, "(%f,%f)\n", p1[i].x - p0[i].x, p1[i].y - p0[i].y);
            good_new.push_back(p1[i]);
            // draw the tracks
            line(mask,p1[i], p0[i], colors[i], 2);
            circle(frame, p1[i], 5, colors[i], -1);
        }else {
            fprintf(fp_opencv, "Feature could not be tracked\n");
        }
    }
    fclose(fp_opencv);
#else
    ownOpticalFlowLK(old_gray, frame_gray, p0, p1, status, err, Size(7,7), 2, criteria);

    FILE *fp_ownOpticalFLow;
    fp_ownOpticalFLow = fopen("own_optical_flow_output.txt", "w+");
    if(fp_ownOpticalFLow == NULL) {
        cout << "Could not open OpenCV Optical Flow file" << endl;
        return -1;
    }


    vector<Point2f> good_new;
    for(uint i = 0; i < p0.size(); i++)
    {
        fprintf(fp_ownOpticalFLow, "Feature Point%d : (%f,%f) => ", i+1, p0[i].x, p0[i].y);
        // Select good points
        if(status[i] == 1) {
            fprintf(fp_ownOpticalFLow, "(%f,%f)\n", p1[i].x - p0[i].x, p1[i].y - p0[i].y);
            good_new.push_back(p1[i]);
            // draw the tracks
            line(mask,p1[i], p0[i], colors[i], 2);
            circle(frame, p1[i], 5, colors[i], -1);
        }else {
            fprintf(fp_ownOpticalFLow, "Feature could not be tracked\n");
        }
    }
    fclose(fp_ownOpticalFLow);

#endif


    //PLot the optical Flow
    /*Mat img;
    add(frame, mask, img);
    namedWindow( "Display window", WINDOW_AUTOSIZE );
    imshow("Display window", img);
    waitKey(0);
    */
    return 0;
}