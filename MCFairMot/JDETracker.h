#pragma once
#include "STrack.h"

class JDETracker
{
public:
	JDETracker() = delete;
	explicit JDETracker(JDETrackerConfig &config, int frame_rate = 30);
	virtual ~JDETracker();
	JDETracker(const JDETracker&) = delete;
	JDETracker& operator=(const JDETracker&) = delete;

public:
	std::map<int, std::vector<std::shared_ptr<STrack>>> UpdateTracking(std::map<int, std::vector<DetectionBox>>& dets, std::map<int, std::vector<cv::Mat>>& id_feature);

private:
	std::tuple<std::vector<std::shared_ptr<STrack>>, std::vector<std::shared_ptr<STrack>>> removeDuplicateStracks(std::vector<std::shared_ptr<STrack>>& stracksa, std::vector<std::shared_ptr<STrack>>& stracksb);
	std::vector<std::shared_ptr<STrack>> subStracks(std::vector<std::shared_ptr<STrack>>& tlista, std::vector<std::shared_ptr<STrack>>& tlistb);
	std::vector<std::shared_ptr<STrack>> jointStracks(std::vector<std::shared_ptr<STrack>>& tlista, std::vector<std::shared_ptr<STrack>>& tlistb);
private:
	JDETrackerConfig& opt_;

	std::map<int, std::vector<std::shared_ptr<STrack>>> tracked_stracks_;
	std::map<int, std::vector<std::shared_ptr<STrack>>> lost_stracks_;
	std::map<int, std::vector<std::shared_ptr<STrack>>> removed_stracks_;

	int frame_id_;
	float det_thresh_;
	int buffer_size_;
	int max_time_lost_;
	int num_class_;

	std::shared_ptr<KalmanFilterTracking> kal_filter_;
};