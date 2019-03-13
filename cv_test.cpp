#include <vector>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <algorithm>


typedef std::vector<cv::Mat> IMG;

void test(const cv::Mat& f_inCvMat, float* vec36)
{
	std::memmove(vec36, f_inCvMat.data, f_inCvMat.total() * f_inCvMat.channels() * sizeof(float));
}

int main() {	
	// 1. copyTo testing
	cv::Mat koala = cv::imread("C:\\Users\\wja4hi\\Desktop\\koala.jpg");
	koala.convertTo(koala, CV_32F, 1 / 255.f);
	cv::Mat paddingmat(koala.rows * 2, koala.cols * 2, CV_32FC3, cvScalar(0, 0, 0));
	koala.copyTo(paddingmat.rowRange(koala.rows, koala.rows * 2).colRange(koala.cols, koala.cols * 2));
	std::cout << koala.depth() << std::endl;

	cv::imshow("result", paddingmat.rowRange(100, 200));
	cv::waitKey(0);

	// 2. image gridding testing	
	cv::Mat blank = cv::Mat::zeros(cv::Size(33, 128), CV_32FC1);
	cv::Size _tmpl_sz(blank.cols, blank.rows);
	std::cout << _tmpl_sz.width << "  " << _tmpl_sz.height << std::endl;
	unsigned _cell_size = 4;
	
	// in horizontal(width) direction, to see how many cells can be set when exceeding the image size is not allowed by (int) 
	if (int(_tmpl_sz.width / (_cell_size)) % 2 == 0) 
	{
		_tmpl_sz.width = (((int)(_tmpl_sz.width / (2 * _cell_size))) * 2 * _cell_size) + _cell_size * 2;
		_tmpl_sz.height = (((int)(_tmpl_sz.height / (2 * _cell_size))) * 2 * _cell_size) + _cell_size * 2;
	}
	else
	{
		_tmpl_sz.width = (((int)(_tmpl_sz.width / (2 * _cell_size))) * 2 * _cell_size) + _cell_size * 3;
		_tmpl_sz.height = (((int)(_tmpl_sz.height / (2 * _cell_size))) * 2 * _cell_size) + _cell_size * 3;
	}

	std::cout << _tmpl_sz.width << "  " << _tmpl_sz.height << std::endl;
	std::cout << int(33 / 4) << std::endl;

	// 3. filter2D testing	
	std::vector<cv::Mat> temp(36, cv::Mat(1, 1, CV_32F));
	cv::Mat src, dst_dy;
	src = cv::imread("C:\\Users\\wja4hi\\Desktop\\koala.jpg");
	std::cout << src.type() << std::endl;
	cv::Mat dst_dx(src.rows, src.cols, CV_32FC3);

	float kernel[3] = { -1, 0, 1 };
	cv::Mat kernel_dy(3, 1, CV_32F, kernel);
	cv::Mat kernel_dx(1, 3, CV_32F, kernel);
	cv::Point anchor = (-1, -1);
	double delta = 0;
	int ddepth = -1;

	if (!src.data) return -1;

	cv::filter2D(src, dst_dx, ddepth, kernel_dx, anchor);
	cv::filter2D(src, dst_dy, ddepth, kernel_dy, anchor);

	std::vector<cv::Mat> channels(3);
	cv::split(src, channels);
	
	cv::imshow("result", channels[0]);
	cv::waitKey(0);

	// 4. IplImage filter testing
	IplImage * dx, * dy, * image;
	image = cvLoadImage("C:\\Users\\wja4hi\\Desktop\\koala.jpg");
	float kernel[3] = { -1.0f, 0.0f, 1.0f};
	CvMat kernel_dx = cvMat(1, 3, CV_32F, kernel);
	CvMat kernel_dy = cvMat(3, 1, CV_32F, kernel);

	dx = cvCreateImage(cvSize(image->width, image->height), IPL_DEPTH_32F, 3);
	dy = cvCreateImage(cvSize(image->width, image->height), IPL_DEPTH_32F, 3);

	cvFilter2D(image, dx, &kernel_dx, cvPoint(-1, 0));
	cvFilter2D(image, dy, &kernel_dy, cvPoint(0, -1));

	std::cout << dx->nChannels << std::endl;

	cvNamedWindow("image");
	cvShowImage("iamge", dx);
	cvWaitKey(0);

	cvReleaseImage(&dx);
	cvReleaseImage(&dy);   
	cvReleaseImage(&image);
	cvDestroyAllWindows();

	// 5. std::vector and math/algorithm testing
	std::vector<int> a(6, 666);
	std::cout << a.size() << std::endl;
	std::cout << a[0] << std::endl;
	std::vector<int>::iterator it = a.begin();
	it = a.insert(it, 200);
	a.insert(it, 6, 300);
	it = a.begin();

	std::vector<int> b(3, 1);
	a.insert(it + 3, b.begin(), b.end());

	//int d[] = {99, 199, 299};
	//a.insert(a.end(), d, d + 3);
	std::vector<std::vector<int>> m(1, b);
	std::vector<int> d({ 99, 199, 299 });
	m.push_back(d);
	std::vector<std::vector<int>>::iterator i;
	for (i = m.begin(); i < m.end(); i++) {
		//std::cout << *i << " ";
	}
	std::cout << "\n";

	for (it = a.begin(); it < a.end(); it++) {
		std::cout << *it << " ";
	}
	std::cout << "\n";


	std::vector<std::vector<int>> c(10, a);
	std::cout << c[0][1] << std::endl;

	int smaller = std::min(3, 666);
	int bigger  = std::max(3, 666);
	std::cout << smaller << " " << bigger << std::endl;

	// 6. opencv general testing
	cv::Point2f poss(5.0f, 10.0f);
	cv::Point pos(poss.operator cv::Point());
	std::cout << pos.x << " " << pos.y << std::endl;

	// 7. c++ general testing
	uchar a = 56;
	std::cout << floor(100 * 1.0f / 3.0f) << std::endl;
	std::cout << a - uchar(2) << std::endl;
	std::cout << int(a - uchar(2)) << std::endl;
	std::cout << (int)'8' << std::endl;
	std::cout << (unsigned char)56 << std::endl;
	std::cout << floor(2.8) << " " << ceil(2.8) << std::endl;
	std::cout << (int)5.8 << std::endl;
	std::cout << (int)5.3 << std::endl;

	//8. hand gradient
	cv::Mat koala = cv::imread("C:\\Users\\wja4hi\\Desktop\\Koala.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	std::cout << "koala.size: " << koala.rows << " " << koala.cols << std::endl;

	cv::Mat koala_flip;
	cv::flip(koala.colRange(0, koala.cols - 1), koala_flip, -1);
	std::cout << "koala flip size: " << koala_flip.rows << " " << koala_flip.cols << std::endl;
	cv::imshow("koala flip", koala_flip);
	cv::waitKey(0);

	cv::Mat cropped_koala(koala.rows / 2 + 1, koala.cols / 2, koala.type());
	cv::Mat cp_koala;
	koala.copyTo(cp_koala);
	cp_koala = cropped_koala;
	std::cout << "cp_koala.size: " << cp_koala.rows << " " << cp_koala.cols << std::endl;
	cv::imshow("cropped koala", cp_koala);
	cv::waitKey(0);

	cv::Mat S, V, D;
	//cv::Mat A = cv::Mat::eye(10, 10, CV_32FC1);
	float demo[16] = { 4, 1, -2, 2, 1, 2, 0, 1, -2, 0, 3, -2, 2, 1, -2, -1};
	//cv::Mat A(4, 4, CV_32FC1, demo);
	cv::Mat A = cv::Mat::eye(4, 4, CV_32FC1);
	cv::SVD::compute(A, S, V, D);
	std::cout << "V is: " << std::endl << V << std::endl;
	std::cout << "D is: " << std::endl << D << std::endl;
	std::cout << "S size: " << S.rows << " " << S.cols << std::endl;
	std::cout << "V size: " << V.rows << " " << V.cols << std::endl;
	std::cout << "D size: " << D.rows << " " << D.cols << std::endl;
	std::cout << "V == D: " << cv::countNonZero(cv::abs(V) != cv::abs(D.t())) << std::endl;

	cv::Mat gradient = cv::Mat::zeros(koala.rows, koala.cols, CV_8UC1);
	//std::cout << koala.size() << " " << gradient.size() << std::endl;
	for (int i = 1; i < koala.rows - 1; i++) {
		for (int j = 1; j < koala.cols - 1; j++) {
			int dx = int(koala.at<uchar>(i, j + 1) - koala.at<uchar>(i, j - 1));
			int dy = int(koala.at<uchar>(i + 1, j) - koala.at<uchar>(i - 1, j));
			//std::cout << "gradient: " << float(::hypot(dx, dy)) << std::endl;
			gradient.at<uchar>(i, j) = int(::hypot(dx, dy));
		}
	}

	int left = 0, top = 0, right = 0, bottom = 0;
	cv::Mat patch;
	cv::Point2i pos_center(int(koala.cols / 2), int(koala.rows / 2));
	cv::Size2i size_roi(int(koala.cols * 1.5), int(koala.rows * 1.5));

	int x1 = pos_center.x - size_roi.width / 2;
	int y1 = pos_center.y - size_roi.height / 2;
	int x2 = pos_center.x + size_roi.width / 2;
	int y2 = pos_center.y + size_roi.height / 2;

	if (x1 < 0)
	{
		left = -x1;
		x1 = 0;
	}
	if (y1 < 0)
	{
		top = -y1;
		y1 = 0;
	}
	if (x2 > koala.cols)
	{
		right = x2 - koala.cols + size_roi.width % 2;
		x2 = koala.cols;
	}
	else
	{
		x2 += size_roi.width % 2;
	}
	if (y2 > koala.rows)
	{
		bottom = y2 - koala.rows + size_roi.height % 2;
		y2 = koala.rows;
	}
	else
		y2 += size_roi.height % 2;

	cv::copyMakeBorder(koala.rowRange(y1, y2).colRange(x1, x2), patch, top, bottom, left, right, cv::BORDER_CONSTANT);

	cv::imshow("gradient", patch);
	cv::waitKey(0);

	float data[18] = { 6, 3, 1, 6, 3, 1, 6, 3, 1, 6, 3, 1, 58, 59, 1, 98, 99, 1};
	cv::Mat testing(3, 2, CV_32FC3, data);	
	
	float data_testing[20];

	test(testing, data_testing);

	//std::cout << "data_testing is: " << data_testing[18] << std::endl;

	std::vector<int> myvector;
	for (int i = 0; i < 10; i++) myvector.push_back(i);
	std::random_shuffle(myvector.begin(), myvector.end());

	std::cout << "my vector contains before: ";
	for (std::vector<int>::iterator it = myvector.begin(); it != myvector.end(); it++)
		std::cout << " " << *it;
	std::cout << std::endl;

	std::nth_element(myvector.begin(), myvector.begin() + 5, myvector.end());
	std::cout << "my vector contains after: ";
	for (std::vector<int>::iterator it = myvector.begin(); it != myvector.end(); it++)
		std::cout << " " << *it;
	std::cout << std::endl;

 	std::vector<cv::Mat> results;
	cv::split(testing, results);
	
	cv::Mat ims_f, ttt;
	testing.convertTo(ims_f, CV_32FC3);
	std::vector<cv::Mat> temp;
	cv::split(ims_f, temp);
	ims_f.copyTo(ttt);
	std::cout << "ims_f type is: " << ttt.type() << std::endl;	
	std::cout << ims_f.channels() << std::endl;
	std::cout << temp[0] << std::endl;
	std::cout << temp[1] << std::endl;
	std::cout << temp[2] << std::endl;

	int sum = 0;
	sum = cv::sum(testing)[1];
	
	std::cout << results[0] << std::endl;
	std::cout << results[1] << std::endl;
	std::cout << results[2] << std::endl;
	std::cout << "The sum is: " << sum << std::endl;
	std::cout << "tesing.total is: " << testing.total() * testing.channels() << std::endl;

	delete data_testing;
	return 0;
}
