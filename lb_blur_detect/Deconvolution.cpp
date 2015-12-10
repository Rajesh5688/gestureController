#include "Deconvolution.h"
#include <opencv2/highgui/highgui.hpp>

Deconvolution::Deconvolution(){

    _deblurMethod = 0;

    // Boost Bind Plans.
    _dftPlan = boost::bind(&Deconvolution::forwardDft, this, _1);
    _idftPlan = boost::bind(&Deconvolution::inverseDft, this, _1);

    // Opencv Stuffs Spatial domain image matrices
    _inputImageMatrix = cv::Mat();
    _outputImageMatrix = cv::Mat();
    _kernelMatrix = cv::Mat();
    _laplacianMatrix = cv::Mat();
    _outLaplacianMatrix = cv::Mat();

    // Opencv Mat Frequency domain image matrices
    _inputImageFFT = cv::Mat();
    _outputImageFFT = cv::Mat();
    _kernelFFT = cv::Mat();
    _kernelTempFFT = cv::Mat();
    _laplacianMatrixFFT = cv::Mat();

    FocusBlur *focusBlur = new FocusBlur();
    focusBlur->_radius = 1.2;
    focusBlur->_smooth = 1;
    focusBlur->_edgeFeather = 0;
    focusBlur->_correctionStrength = 0;
    _blur = focusBlur;
}

void Deconvolution::initFFT(const cv::Mat inputImage) {
    removeFFTObjects();
    QTime time;
    time.start();
    // Read image size
    _width = inputImage.cols;
    _height = inputImage.rows;

    // Init Opencv Spatial domain image structures with given size
    _inputImageMatrix.create(_height, _width, CV_32FC1);
    _kernelMatrix.create(_height, _width, CV_32FC1);
    _laplacianMatrix.create(_height, _width, CV_32FC1);
    _outLaplacianMatrix.create(_height, _width, CV_32FC1);

    // Init Frequency Domain image structures with given size
    int paddedHeight = cv::getOptimalDFTSize(_height);
    int paddedWidth =  cv::getOptimalDFTSize(_width);

    _outputImageMatrix.create(paddedHeight, paddedWidth, CV_32FC1);
    _inputImageFFT.create(paddedHeight, paddedWidth, CV_32FC2);
    _outputImageFFT.create(paddedHeight, paddedWidth, CV_32FC2);
    _kernelFFT.create(paddedHeight, paddedWidth, CV_32FC2);
    _kernelTempFFT.create(paddedHeight, paddedWidth, CV_32FC2);
    _laplacianMatrixFFT.create(paddedHeight, paddedWidth, CV_32FC2);

    _spatialMap["input"] = _inputImageMatrix;
    _spatialMap["output"] = _outputImageMatrix;
    _spatialMap["laplacian"] = _laplacianMatrix;
    _spatialMap["kernel"] = _kernelMatrix;

    _frequencyMap["input"] = _inputImageFFT;
    _frequencyMap["output"] = _inputImageFFT;
    _frequencyMap["laplacian"] = _laplacianMatrixFFT;
    _frequencyMap["kernel"] = _kernelFFT;

    qDebug() << "Time Taken to initFFT: " << time.elapsed() << "ms.";

}

bool Deconvolution::forwardDft(std::string inputType)
{
    std::cout << inputType << std::endl;
    if(_spatialMap.find(inputType) != _spatialMap.end())
    {
        if(_spatialMap.find(inputType)->second.channels() == 1)
        {
            cv::Mat inputImg = _spatialMap.find(inputType)->second;
            cv::Mat paddedImg;
            cv::copyMakeBorder(inputImg, paddedImg, 0, _frequencyMap.find(inputType)->second.rows - inputImg.rows,
                               0, _frequencyMap.find(inputType)->second.cols - inputImg.cols, cv::BORDER_CONSTANT, cv::Scalar::all(0));

            cv::Mat planes[] = {paddedImg, cv::Mat::zeros(paddedImg.size(), CV_32F)};
            cv::Mat complex;
            cv::merge(planes, 2, complex);
            cv::dft(complex, _frequencyMap.find(inputType)->second);
        }
        else
        {
            cv::dft(_spatialMap.find(inputType)->second, _frequencyMap.find(inputType)->second);
        }
        return true;
    }
    else
    {
        return false;
    }
}

bool Deconvolution::inverseDft(std::string inputType)
{
    std::cout << inputType << std::endl;
    if(_frequencyMap.find(inputType) != _frequencyMap.end())
    {
        cv::dft(_frequencyMap.find(inputType)->second, _spatialMap.find(inputType)->second, cv::DFT_INVERSE | cv::DFT_REAL_OUTPUT);
        return true;
    }
    else
    {
        return false;
    }
}

Deconvolution::~Deconvolution() {
    removeFFTObjects();
}

void Deconvolution::removeFFTObjects() {

    _inputImageFFT.release();
    _outputImageFFT.release();
    _kernelFFT.release();
    _kernelTempFFT.release();
    _laplacianMatrixFFT.release();

    _isProcessingCancelled = false;
}


bool Deconvolution::doDeconvolution(cv::Mat inputImage, cv::Mat& outputImage) {

    _isProcessingCancelled = false;
    // Create kernel
    buildKernel(_kernelMatrix, _width, _height);
    _dftPlan("kernel");

    // Fill processingContext
    DeconvContext* processingContext = new DeconvContext();
    processingContext->_outputImageMatrix = _outputImageMatrix;
    processingContext->_kernelFFT = _kernelFFT;
    processingContext->_width = _width;
    processingContext->_height = _height;
    processingContext->_blur = _blur;

    cv::Rect roi = cv::Rect(0, 0, _width, _height);
    if (inputImage.channels() == 1) {
        processingContext->_inputImage = inputImage;
        inputImage.convertTo(_inputImageMatrix, CV_32F);
        processingContext->_inputImageMatrix = _inputImageMatrix;
        doDeconvolutionForChannel(processingContext);
        outputImage = getImageFromMatrix(processingContext);
        outputImage = outputImage(roi);
    } else {
        if (inputImage.channels() != 3) {
            qFatal("ERROR in doConvolution: Currently supports only Color Image and Grey scale image.");
        }
        processingContext->_inputImage = inputImage;
        std::vector<cv::Mat> splitChannels;
        cv::split(inputImage, splitChannels);

        for(int eachChannel = 0; eachChannel < 3; eachChannel++)
        {
            splitChannels[eachChannel].convertTo(_inputImageMatrix, CV_32F);
            processingContext->_inputImageMatrix = _inputImageMatrix;
            doDeconvolutionForChannel(processingContext);
            cv::Mat debluredImg = getImageFromMatrix(processingContext);
            debluredImg.convertTo(debluredImg, CV_8U);
            splitChannels[eachChannel] = debluredImg(roi);
        }
        cv::merge(splitChannels, outputImage);
    }
    inputImage.convertTo(inputImage, CV_8U);

    delete(processingContext);
    return !_isProcessingCancelled;
}

void Deconvolution::buildKernel(cv::Mat &outKernel, const int width, const int height) {

    QImage* kernelImage;
    double* kernelTempMatrix = (double*)malloc(sizeof(double)*width*height);
    kernelImage = _blur->buildKernelImage();

    int size = kernelImage->width();
    // Fill kernel
    double sumKernelElements = 0;
    for (int y = 0; y<height; y++) {
        for (int x = 0; x<width; x++) {
            int index = y*width + x;
            int value = 0;
            // if we are in the kernel area (of small kernelImage), then take pixel values. Otherwise keep 0
            if (abs(x-width/2)<(size-2)/2 && abs(y-height/2)<(size-2)/2) {
                int xLocal = x-(width-size)/2;
                int yLocal = y-(height-size)/2;
                value = qRed(kernelImage->pixel(xLocal,yLocal));
            }
            kernelTempMatrix[index] = value;
            sumKernelElements += abs(value);
        }
    }
    delete(kernelImage);
    // Zero-protection
    if (sumKernelElements==0) {
        sumKernelElements = 1;
    }
    // Normalize
    double k = 1/sumKernelElements;
    for (int i=0; i<width*height; i++) {
        kernelTempMatrix[i] *= k;
    }
    // Translate kernel, because we don't use centered FFT (by multiply input image on pow(-1,x+y))
    // so we need to translate kernel by width/2 to the left and by height/2 to the up
    for (int y=0; y<height; y++) {
        float* ptr = (float *)(outKernel.ptr<float>(y));
        for (int x=0; x<width; x++) {
            int xTranslated = (x + width/2) % width;
            int yTranslated = (y + height/2) % height;
            ptr[x] = kernelTempMatrix[yTranslated*width + xTranslated];
        }
    }
    free(kernelTempMatrix);
}

void Deconvolution::doDeconvolutionForChannel(DeconvContext* processingContext) {

    double blurRadius = processingContext->_blur->_radius;
    int width = processingContext->_width;
    int height = processingContext->_height;

    _dftPlan("input");
    processingContext->_inputImageFFT = _inputImageFFT;
    // Borders processing to prevent ring effect
    multiplyRealFFTs(_inputImageFFT, processingContext->_kernelFFT, _inputImageFFT.cols, _inputImageFFT.rows);
    _idftPlan("output");

    for (int y = 0; y<height; y++) {
        float* ptrOut = _outputImageMatrix.ptr<float>(y);
        float* ptrIn = processingContext->_inputImageMatrix.ptr<float>(y);
        for (int x = 0; x<width; x++) {
            if (x < blurRadius || y < blurRadius || x > width - blurRadius ||y > height - blurRadius) {
                ptrIn[x] = ptrOut[x] / (width * height);
            }
        }
    }
    // Deconvolution in the Frequency domain
    _dftPlan("input");
    if (_deblurMethod == 0) {
        deconvolutionByWiener(processingContext);
    } else {
        deconvolutionByTikhonov(processingContext);
    }
    _idftPlan("output");
}

void Deconvolution::multiplyRealFFTs(cv::Mat& outFFT, const cv::Mat kernelFFT, const int width, const int height) {
    for (int y = 0; y<height; y++) {
        for (int x = 0; x<width; x++) {
            cv::Point_<float>* row_ptr = outFFT.ptr<cv::Point_<float> >(y,x);
            const cv::Point_<const float>* kernel_ptr = kernelFFT.ptr<cv::Point_<const float> >(y,x);
            double value = kernel_ptr->x;
            row_ptr->x *= value;
            row_ptr->y *= value;
        }
    }
}

void Deconvolution::deconvolutionByWiener(DeconvContext* processingContext) {

    double K = pow(1.07, processingContext->_blur->_smooth)/10000.0;
    for(int y = 0; y < processingContext->_inputImageFFT.rows; ++y)
    {
        for(int x = 0; x < processingContext->_inputImageFFT.cols; ++x)
        {
            cv::Point_<float>* ptrInputImg = processingContext->_inputImageFFT.ptr<cv::Point_<float> >(y,x);
            cv::Point_<float>* ptrKernel = processingContext->_kernelFFT.ptr<cv::Point_<float> >(y,x);

            double energyValue = pow(ptrKernel->x, 2) + pow(ptrKernel->y, 2);
            double wienerValue = ptrKernel->x / (energyValue + K);
            ptrInputImg->x = wienerValue * ptrInputImg->x;
            ptrInputImg->y = wienerValue * ptrInputImg->y;
        }
    }
}

void Deconvolution::deconvolutionByTikhonov(DeconvContext* processingContext) {

    // Create laplacian
    cv::Mat kx, ky;
    cv::getDerivKernels(kx, ky, 2, 2, 3, true, CV_32FC1);
    _laplacianMatrix = kx * ky.t();
    _dftPlan("laplacian");

    double K = pow(1.07, processingContext->_blur->_smooth)/1000.0;

    for(int y = 0; y < processingContext->_inputImageFFT.rows; ++y)
    {
        for(int x = 0; x < processingContext->_inputImageFFT.cols; ++x)
        {
            cv::Point_<float>* ptrInputImg = processingContext->_inputImageFFT.ptr<cv::Point_<float> >(y,x);
            cv::Point_<float>* ptrKernel = processingContext->_kernelFFT.ptr<cv::Point_<float> >(y,x);
            cv::Point_<float>* ptrLaplace = _laplacianMatrixFFT.ptr<cv::Point_<float> >(y,x);

            double energyValue = pow(ptrKernel->x, 2) + pow(ptrKernel->y, 2);
            double energyLaplacianValue = pow(ptrLaplace->x, 2) + pow(ptrLaplace->y, 2);
            double tikhonovValue = ptrKernel->x / (energyValue + K*energyLaplacianValue);
            ptrInputImg->x = tikhonovValue * ptrInputImg->x;
            ptrInputImg->y = tikhonovValue * ptrInputImg->y;
        }
    }
}

void Deconvolution::visualizeFFT(cv::Mat fft, QString path) {
    std::vector<cv::Mat> planes;
    cv::split(fft, planes);                   // planes[0] = Re(DFT(I), planes[1] = Im(DFT(I))
    cv::magnitude(planes[0], planes[1], planes[0]);// planes[0] = magnitude
    cv::Mat magI = planes[0];

    // switch to logarithmic scale
    magI += cv::Scalar::all(1);
    log(magI, magI);

    // crop the spectrum, if it has an odd number of rows or columns
    magI = magI(cv::Rect(0, 0, magI.cols & -2, magI.rows & -2));

    // rearrange the quadrants of Fourier image  so that the origin is at the image center
    int cx = magI.cols/2;
    int cy = magI.rows/2;

    // Steps to center the specturm signal, going from low frequency at the center then going out.
    cv::Mat q0(magI, cv::Rect(0, 0, cx, cy));
    cv::Mat q1(magI, cv::Rect(cx, 0, cx, cy));
    cv::Mat q2(magI, cv::Rect(0, cy, cx, cy));
    cv::Mat q3(magI, cv::Rect(cx, cy, cx, cy));

    cv::Mat tmp;
    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);

    q1.copyTo(tmp);
    q2.copyTo(q1);
    tmp.copyTo(q2);

    if(!path.isEmpty())
    {
        cv::imwrite(path.toStdString(), magI);
    }
}

cv::Mat Deconvolution::getImageFromMatrix(DeconvContext *processingContext)
{    
    cv::Mat outputImage;
    outputImage.create(processingContext->_outputImageMatrix.rows, processingContext->_outputImageMatrix.cols, CV_8UC1);
    double k = 1.0/(processingContext->_outputImageMatrix.cols * processingContext->_outputImageMatrix.rows);
    for (int y = 0; y < processingContext->_height; y++) {
        float* ptr = processingContext->_outputImageMatrix.ptr<float>(y);
        uchar* ptr_result = outputImage.ptr<uchar>(y);
        for (int x = 0; x < processingContext->_width; x++) {
            double value = k * ptr[x];
            ptr_result[x] =  int(value);
        }
    }
    return outputImage;
}


