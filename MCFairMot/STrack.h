#pragma once
#include "Config.h"
#include "Kalmanfilter.h"
#include "DataType.h"

enum struct TrackState :unsigned char { New, Tracked, Lost, Removed };

class STrack
{
public:
	STrack() = delete;
	explicit STrack(cv::Rect_<float>& tlwh, float score, cv::Mat temp_feat, int class_id);
	virtual ~STrack();
	STrack(const STrack&) = delete;
	STrack& operator=(const STrack&) = delete;
public:
	void Predict();
	void Activate(std::shared_ptr<KalmanFilterTracking> kal_filter, int frame_id, int track_id);
	void Reactivate(std::shared_ptr<STrack> new_track, int frame_id, int track_id, bool new_id = false);
	void Update(std::shared_ptr<STrack> new_track, int frame_id, bool update_feature = true);
	DETECTBOX TlwhToBox();
	cv::Rect_<float> TlwhToRect();
	DETECTBOX TlwhToXyah(const cv::Rect_<float>& tlwh);

	//static int NextTrackID(int cls_id);
	//static void ResetTrackID(int cls_id);
	//static  std::map<int, int> InitTrackCount(int num_cls);

private:
	void updateFeatures(cv::Mat& feat);

private:
	//static std::map<int, int> _track_count;

	int _tracklet_len = 0;
	float _alpha = 0.9f;
	int _class_id = 0;
	int _time_since_update = 0;

	int _track_id = 0;
	float _det_score = 0;
	int _frame_id = 0;
	int _start_frame = 0;
	cv::Mat _curr_feat;
	cv::Rect_<float> _tlwh;
	TrackState _track_state = TrackState::New;
	std::shared_ptr<KalmanFilterTracking> _kal_filter;

	KAL_MEAN _mean;
	KAL_COVA _covariance;
	bool _is_activated = false;
	cv::Mat _smooth_feat;

	std::vector<cv::Rect2f> _trajectory;
public:
	inline int GetFrameID() {
		return _frame_id;
	}
	inline int GetStartFrame() {
		return _start_frame;
	}
	inline int GetTrackID() {
		return _track_id;
	}
	inline float GetDetScore() {
		return _det_score;
	}
	inline void MarkLost()
	{
		_track_state = TrackState::Lost;
	}
	inline void MarkRemove()
	{
		_track_state = TrackState::Removed;
	}
	inline TrackState GetTrackState()
	{
		return _track_state;
	}
	inline cv::Mat GetCurrentFeat() {
		return _curr_feat;
	}
	inline KAL_MEAN GetMean() {
		return _mean;
	}
	inline KAL_COVA GetCovariance() {
		return _covariance;
	}
	inline bool IsActivated() {
		return _is_activated;
	}
	inline cv::Mat GetSmoothFeat() {
		return _smooth_feat;
	}
	inline std::vector<cv::Rect2f> GetTrajectory() {
		return _trajectory;
	}
};
