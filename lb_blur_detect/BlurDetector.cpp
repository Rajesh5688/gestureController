#include "BlurDetector.h"

/** Only for Internal Use.**/
namespace BlurDetectorDebug {
bool debugFlag= false;
}

BlurDetector::BlurDetector()
{
    _deconv = NULL;
}

double BlurDetector::detectBluriness(cv::Mat img, int detectionAlgorithm)
{
    double rtn = 0.0;
    if(detectionAlgorithm == LAPM)
    {
        rtn = modifiedLaplacian(img);
    }
    if(detectionAlgorithm == LAPV)
    {
        rtn = varianceOfLaplacian(img);
    }
    if(detectionAlgorithm == GRAD)
    {
        rtn = sobelGradients(img, 5);
    }
    if(detectionAlgorithm == GAUSSV)
    {
        rtn = varianceOfGaussian(img);
    }
    if(detectionAlgorithm == MEDV)
    {
        rtn = varianceOfMedian(img);
    }
    if(detectionAlgorithm == DEBLUR)
    {
        if(_deconv == NULL)
        {
            _deconv = new Deconvolution();
            _deconv->initFFT(img);
            rtn = deblurResponse(img);
        }
        else
        {
            rtn = deblurResponse(img);
        }
    }

    return rtn;
}

double BlurDetector::deblurResponse(const cv::Mat img_)
{
    cv::Mat src_grey;
    cv::cvtColor( img_, src_grey, CV_RGB2GRAY );

    cv::Mat mask = maskSaturatedPixels(src_grey);
    mask = mask/255;
    cv::multiply(src_grey, mask, src_grey);

    cv::Mat outputImage;
    _deconv->doDeconvolution(src_grey, outputImage);

    double mean = BRT::IMG_STATISTICS::getMean( (src_grey - outputImage), mask);
    double sigma = BRT::IMG_STATISTICS::getSquareOfVariance((src_grey-outputImage), mask);

    if(BlurDetectorDebug::debugFlag)
    {
        qDebug() << " Variance : " << sigma << " , Mean : " << mean;
        cv::imshow("Img", src_grey);
        cv::imshow("Deblured Img", outputImage);
        cv::waitKey(0);
    }

    return sigma*mean;
}

// 'LAPM' algorithm (Nayar89)
double BlurDetector::modifiedLaplacian(const cv::Mat src)
{
    cv::Mat M = (cv::Mat_<double>(5, 1) << -1, -2, 6, -2, -1);
    cv::Mat G = cv::getGaussianKernel(5, -1, CV_64F);

    cv::Mat preProcessedSrc = preprocessImage(src, cv::Mat());
    cv::Mat src_grey;
    cvtColor( preProcessedSrc, src_grey, CV_RGB2GRAY );
    src_grey.convertTo(src_grey, CV_64F);

    cv::Mat Lx;
    cv::sepFilter2D(src_grey, Lx, CV_64F, M, G);

    cv::Mat Ly;
    cv::sepFilter2D(src_grey, Ly, CV_64F, G, M);

    cv::Mat FM = cv::abs(Lx) + cv::abs(Ly);

    double focusMeasure = BRT::IMG_STATISTICS::getMean(FM);
    return focusMeasure;
}

cv::Mat BlurDetector::preprocessImage(const cv::Mat img, const cv::Mat mask)
{
    cv::Mat rtn;
    if(img.channels() == 3)
    {
        rtn.create(img.rows,img.cols, CV_32FC3);
        img.convertTo(rtn, CV_32FC3);
        std::vector<cv::Mat> splitChannels;
        cv::split(rtn, splitChannels);

        // Make the data zero-centered.
        for(int eachChannel = 0; eachChannel < 3; eachChannel++)
        {
            splitChannels[eachChannel] = splitChannels[eachChannel] - BRT::IMG_STATISTICS::getMean(splitChannels[eachChannel], mask);
            splitChannels[eachChannel] = splitChannels[eachChannel] / BRT::IMG_STATISTICS::getVariance(splitChannels[eachChannel], mask);
        }
        cv::merge(splitChannels, rtn);
    }
    else if(img.channels() == 1)
    {
        img.convertTo(rtn, CV_32FC1);
        rtn = rtn - BRT::IMG_STATISTICS::getMean(img, mask);
        rtn = rtn / BRT::IMG_STATISTICS::getVariance(img, mask);
    }
    return rtn;
}

double BlurDetector::varianceOfMedian(const cv::Mat src)
{
    QTime time;
    time.start();

    cv::Mat src_grey;
    if( src.channels() == 3 ){
        cvtColor( src, src_grey, CV_RGB2GRAY );
    }
    else {
        src_grey = src;
    }
    cv::Mat mask = maskSaturatedPixels(src_grey);

    // Get a filter size that is 2% of image dimension.
    int kernelSize = (((int)((double)src_grey.cols * 0.02) % 2)== 0)? (int) (src_grey.cols*0.02+1) : (src_grey.cols*0.02);
    cv::Mat medianFiltered;
    cv::medianBlur(src_grey, medianFiltered, kernelSize);

    double blurMeasure = BRT::IMG_STATISTICS::getSquareOfVariance(src_grey-medianFiltered, mask);
    double timeElapsed = time.elapsed();

    if(BlurDetectorDebug::debugFlag)
    {
        qDebug() << " Difference Mean : " << BRT::IMG_STATISTICS::getMean(src_grey-medianFiltered, mask)
                 << " , Variance : " << BRT::IMG_STATISTICS::getVariance(src_grey-medianFiltered, mask);
        qDebug() << "Elapsed : " << timeElapsed;
        qDebug() << "\n \n";

        cv::imshow("Img Original", src);
        cv::imshow("Mask", mask);
        cv::waitKey(0);
    }

    return blurMeasure;
}

double BlurDetector::varianceOfGaussian(const cv::Mat src)
{
    QTime time;
    time.start();

    cv::Mat src_grey;
    if( src.channels() == 3 ){
        cvtColor( src, src_grey, CV_RGB2GRAY );
    }
    else {
        src_grey = src;
    }
    cv::Mat mask = maskSaturatedPixels(src_grey);

    // Get a filter size that is covering 2% percent of image both in width and height.
    int kernelWidth = (((int)((double)src_grey.cols * 0.02) % 2)== 0)? (int) (src_grey.cols*0.02+1) : (src_grey.cols*0.02);
    int kernelHeight = (((int)((double)src_grey.rows * 0.02) % 2)== 0)? (int) (src_grey.rows*0.02+1) : (src_grey.rows*0.02);
    cv::Mat gaussian;
    cv::GaussianBlur(src_grey, gaussian, cv::Size(kernelWidth, kernelHeight), CV_32F);

    double blurMeasure = BRT::IMG_STATISTICS::getSquareOfVariance(src_grey-gaussian, mask);
    double timeElapsed = time.elapsed();
    if(BlurDetectorDebug::debugFlag)
    {
        qDebug() << " Difference Mean : " << BRT::IMG_STATISTICS::getMean(src_grey-gaussian, mask)
                 << " , Variance : " << blurMeasure;

        qDebug() << "Elapsed : " << timeElapsed;
        qDebug() << "\n \n";

        cv::imshow("Img Original", src);
        cv::imshow("Mask", mask);
        cv::waitKey(0);
    }

    return blurMeasure;
}

cv::Mat BlurDetector::maskSaturatedPixels(const cv::Mat img)
{
    if(img.channels() > 1)
    {
        std::cerr << "ERROR In maskSaturatedPixels: Image should be single channel." << std::cout;
    }

    // Mask pixels with 240 or greater intensities with '0'
    //  pelle: change to 100. Lots of noise in saturated samples.
    //  Normal lettuce images are all well below 100.
    int intensityVal = 100;
    cv::Mat mask;
    cv::compare(img, intensityVal, mask, CV_CMP_LT);

    return mask;

}

// 'LAPV' algorithm (Pech2000)
double BlurDetector::varianceOfLaplacian(const cv::Mat src)
{
    QTime time;
    time.start();
    cv::Mat src_grey;
    cvtColor( src, src_grey, CV_RGB2GRAY );

    cv::Mat mask = maskSaturatedPixels(src_grey);
    cv::Mat preProcessedSrc = preprocessImage(src_grey, mask);

    cv::Mat lap;
    // Using second Order Derivative of size 5x5 filter.
    cv::Laplacian(preProcessedSrc, lap, preProcessedSrc.depth(), 5, CV_32F);

    double focusMeasure = BRT::IMG_STATISTICS::getSquareOfVariance(lap, mask);
    double mean = BRT::IMG_STATISTICS::getMean(lap, mask);

    if(BlurDetectorDebug::debugFlag)
    {
        qDebug() << "Variance Measure : " << focusMeasure << " , Mean : " << mean;
        qDebug() << "Elapsed : " << time.elapsed();
        qDebug() << "\n \n";
    }

    return focusMeasure;
}

// 'TENG' algorithm (Krotkov86)
double BlurDetector::sobelGradients(const cv::Mat src, int ksize)
{
    cv::Mat Gx, Gy;
    cv::Sobel(src, Gx, CV_64F, 1, 0, ksize);
    cv::Sobel(src, Gy, CV_64F, 0, 1, ksize);

    cv::Mat FM = Gx.mul(Gx) + Gy.mul(Gy);

    std::vector<cv::Mat> splitChannels;
    cv::split(FM, splitChannels);
    double focusMeasure = 0.0;
    for(int eachChannel=0; eachChannel < src.channels(); eachChannel++)
    {
        focusMeasure += BRT::IMG_STATISTICS::getMean(splitChannels[eachChannel]);
    }

    return focusMeasure/(double)src.channels();
}
