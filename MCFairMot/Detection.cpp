#include <fstream>
//#include "Utils.h"
#include "Detection.h"

static Logger gLogger;


FairMOTDetector::FairMOTDetector(FairMOTDetectorConfig& config)
	:params_(config)
	, frame_idx_(-1)
{

}

FairMOTDetector::~FairMOTDetector()
{
	delete[] data_; data_ = NULL;
	for (int i = 0; i < output_tensor_num_; i++)
	{
		delete[] output_data_[i];
		output_data_[i] = NULL;
	}
	context_->destroy();
	engine_->destroy();
	runtime_->destroy();
}


bool FairMOTDetector::generateEngine(std::string& onnx_model_name, std::string& engine_model_name)
{
	auto builder = nvinfer1::createInferBuilder(gLogger);
	if (!builder)
	{
		return false;
	}

	const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
	nvinfer1::INetworkDefinition *network = builder->createNetworkV2(explicitBatch);

	if (!network)
	{
		return false;
	}

	auto config = builder->createBuilderConfig();
	if (!config)
	{
		return false;
	}

	auto parser = nvonnxparser::createParser(*network, gLogger);
	if (!parser)
	{
		return false;
	}

	parser->parseFromFile(onnx_model_name.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kINFO));
	config->setMaxWorkspaceSize(16 * (1 << 20));  // 16MB
	config->setFlag(nvinfer1::BuilderFlag::kTF32);

	std::cout << "Building engine, please wait for a while..." << std::endl;
	nvinfer1::ICudaEngine* mEngine = builder->buildEngineWithConfig(*network, *config);
	if (!mEngine)
	{
		return false;
	}
	std::cout << "Build engine successfully!" << std::endl;

	auto mInputDims = network->getInput(0)->getDimensions();
	auto mOutputDims = network->getOutput(0)->getDimensions();
	std::cout << "network->getNbOutputs(): " << network->getNbOutputs() << std::endl;
	std::cout << "mOutputDims.nbDims: " << mOutputDims.nbDims << std::endl;

	network->destroy();
	// Serialize the engine
	nvinfer1::IHostMemory* modelStream = mEngine->serialize();

	std::ofstream p(engine_model_name, std::ios::binary);
	if (!p) {
		std::cerr << "could not open plan output file" << std::endl;
		return false;
	}
	p.write(reinterpret_cast<const char*>(modelStream->data()), modelStream->size());
	modelStream->destroy();

	// Close everything down
	mEngine->destroy();
	builder->destroy();
	return true;
}

bool FairMOTDetector::InitDetection(bool is_generate_engine)
{
	auto engine_model_name = params_.model_file + ".fp32.trtmodel";
	if (is_generate_engine)
	{
		auto onnx_model_name = params_.model_file + ".onnx";
		if (onnx_model_name == "")
		{
			printf("onnx models not found");
			return false;
		}

		if (!generateEngine(onnx_model_name, engine_model_name))
		{
			printf("convert onnx models failed");
			return false;
		}
	}

	char*trtModelStream; int size;
	std::ifstream file(engine_model_name, std::ios::binary);
	if (file.good())
	{
		file.seekg(0, file.end);
		size = file.tellg();
		file.seekg(0, file.beg);
		trtModelStream = new char[size];
		file.read(trtModelStream, size);
		file.close();
	}
	else return false;

	runtime_ = nvinfer1::createInferRuntime(gLogger);
	assert(runtime_ != nullptr);
	engine_ = runtime_->deserializeCudaEngine(trtModelStream, size);
	assert(engine_ != nullptr);
	context_ = engine_->createExecutionContext();
	assert(context_ != nullptr);
	delete[] trtModelStream;
	const nvinfer1::ICudaEngine& engine = context_->getEngine();
	input_index_ = engine.getBindingIndex(input_name_.c_str());
	auto input_dims = engine.getBindingDimensions(input_index_);
	input_h_ = input_dims.d[2];
	input_w_ = input_dims.d[3];
	data_ = new float[3 * input_h_ * input_w_];

	for (int i = 0; i < output_tensor_num_; i++)
	{
		output_idx_[i] = engine.getBindingIndex(output_names_[i].c_str());
		auto output_dims = engine.getBindingDimensions(output_idx_[i]);
		int size = 1;
		for (int j = 0; j < output_dims.nbDims; j++)
		{
			size *= output_dims.d[j];
		}
		output_data_[i] = new float[size];
		output_size_[i] = size;
	}
	return true;
}

cv::Rect FairMOTDetector::restoreCenterNetBox(float dx, float dy, float l, float t, float r, float b, float cellx, float celly, int stride, cv::Size netSize, cv::Size imageSize)
{
	float scale = 0;
	//if(imageSize.width >= imageSize.height)
	if ((float(netSize.width) / float(imageSize.width)) <= (float(netSize.height) / float(imageSize.height))) 
		scale = netSize.width / (float)imageSize.width;
	else
		scale = netSize.height / (float)imageSize.height;

	float xx = ((cellx + dx - l) * stride) / scale ;
	float yy = ((celly + dy - t) * stride) / scale ;
	float rr = ((cellx + dx + r) * stride) / scale;
	float bb = ((celly + dy + b) * stride) / scale;
	return cv::Rect(cv::Point(xx, yy), cv::Point(rr + 1, bb + 1));
}

cv::Rect FairMOTDetector::restoreCenterNetBox(float dx, float dy, float w, float h, float cellx, float celly, int stride, cv::Size netSize, cv::Size imageSize)
{
	float scale = 0;
	//if(imageSize.width >= imageSize.height)
	if ((float(netSize.width) / float(imageSize.width)) <= (float(netSize.height) / float(imageSize.height)))
		scale = netSize.width / (float)imageSize.width;
	else
		scale = netSize.height / (float)imageSize.height;

	float xx = ((cellx + dx - w*0.5) * stride) / scale;
	float yy = ((celly + dy - h*0.5) * stride) / scale;
	float rr = ((cellx + dx + w*0.5) * stride) / scale;
	float bb = ((celly + dy + h*0.5) * stride) / scale;
	return cv::Rect(cv::Point(xx, yy), cv::Point(rr + 1, bb + 1));
}



void FairMOTDetector::GetDetection(cv::Mat& frame, int cls_num, std::map<int, std::vector<DetectionBox>>& vec_db, std::map<int, std::vector<cv::Mat>>& vec_features)
{
	vec_db.clear();
	vec_features.clear();
	frame_idx_++;

	int frame_width = frame.cols;
	int frame_height = frame.rows;
	cv::Mat sized; int padding = 0;
	std::vector<cv::Mat> ms(frame.channels());

	float ratio = (float(input_w_) / float(frame_width)) < (float(input_h_) / float(frame_height))? (float(input_w_) / float(frame_width)) : (float(input_h_) / float(frame_height));
	cv::Mat flt_img = cv::Mat::zeros(cv::Size(input_w_, input_h_), CV_8UC3);
	cv::Mat rsz_img;
	cv::resize(frame, rsz_img, cv::Size(), ratio, ratio);
	rsz_img.copyTo(flt_img(cv::Rect(0, 0, rsz_img.cols, rsz_img.rows)));
	flt_img.convertTo(flt_img, CV_32FC3, 1.0 / 255);
	for (int i = 0; i < ms.size(); ++i)
		ms[i] = cv::Mat(input_h_, input_w_, CV_32F, &data_[i*input_h_ * input_w_]);
	cv::split(flt_img, ms);
	const nvinfer1::ICudaEngine& engine = context_->getEngine();
	void*buffers[output_tensor_num_ + 1];
	int inputIndex = 0;
	auto input_dims = engine.getBindingDimensions(inputIndex);
	int input_size = 1;
	for (int j = 0; j < input_dims.nbDims; j++)
	{
		input_size *= input_dims.d[j];
	}
	CHECK(cudaMalloc(&buffers[inputIndex], input_size * sizeof(float)));

	for (int i = 0; i < output_tensor_num_; i++)
	{
		CHECK(cudaMalloc(&buffers[output_idx_[i]], output_size_[i] * sizeof(float)));
	}
	// Create stream
	cudaStream_t stream;
	CHECK(cudaStreamCreate(&stream));

	// DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
	CHECK(cudaMemcpyAsync(buffers[inputIndex], data_, input_size * sizeof(float), cudaMemcpyHostToDevice, stream));
	context_->enqueue(1, buffers, stream, nullptr);
	for (int i = 0; i < output_tensor_num_; i++)
	{
		CHECK(cudaMemcpyAsync(output_data_[i], buffers[output_idx_[i]], output_size_[i] * sizeof(float), cudaMemcpyDeviceToHost, stream));
	}
	cudaStreamSynchronize(stream);

	// Release stream and buffers
	cudaStreamDestroy(stream);
	CHECK(cudaFree(buffers[inputIndex]));
	for (int i = 0; i < output_tensor_num_; i++)
	{
		CHECK(cudaFree(buffers[output_idx_[i]]));
	}

	float *hm_ptr = output_data_[0];
	float *wh_ptr = output_data_[1];
	float *reg_ptr = output_data_[2];
	float *hmpool_ptr = output_data_[3];
	float *id_ptr = output_data_[4];
	int size = output_size_[0];
	int x, y;
	int down_ratio = 4;
	int hm_width = engine.getBindingDimensions(output_idx_[0]).d[3];
	int hm_height = engine.getBindingDimensions(output_idx_[0]).d[2];
	int id_dims[4];
	for (int i = 0; i < 4; i++) id_dims[i] = engine.getBindingDimensions(output_idx_[4]).d[i];

	int outsize = hm_width * hm_height;
	for (int cls = 0; cls < cls_num; cls++) {
		float* hm_ptr_cls = hm_ptr + outsize * cls;
		float* hmpool_ptr_cls = hmpool_ptr + outsize * cls;
		for (int idx = 0; idx < outsize; idx++)
		{
			if (*hm_ptr_cls == *hmpool_ptr_cls && *hmpool_ptr_cls > params_.threshold)
			{
				x = idx % hm_width;
				y = idx / hm_width;

				float dx = *(reg_ptr + idx);
				float dy = *(reg_ptr + outsize + idx);

				//float l = *(wh_ptr + idx);
				//float t = *(wh_ptr + outsize + idx);
				//float r = *(wh_ptr + outsize * 2 + idx);
				//float b = *(wh_ptr + outsize * 3 + idx);

				//cv::Rect box = restoreCenterNetBox(dx, dy, l, t, r, b, x, y, down_ratio, cv::Size(input_w, input_h), frame.size());

				float w = *(wh_ptr + idx);
				float h = *(wh_ptr + outsize + idx);

				cv::Rect box = restoreCenterNetBox(dx, dy, w, h, x, y, down_ratio, cv::Size(input_w_, input_h_), frame.size());

				box = box & cv::Rect(0, 0, frame.cols, frame.rows);
				if (box.area() > 0)
				{
					vec_db[cls].push_back(DetectionBox(frame_idx_, cv::Rect_<float>(box.x, box.y, box.width, box.height), *hmpool_ptr_cls));
					float* dim_fea = id_ptr + y * (id_dims[2] * id_dims[3]) + x * id_dims[3];
					cv::Mat fea(1, id_dims[3], CV_32FC1, dim_fea);
					float* tmpdata2 = (float*)fea.data;
					vec_features[cls].emplace_back(fea.clone());
				}
			}
			hm_ptr_cls++;
			hmpool_ptr_cls++;
		}
	}

}
