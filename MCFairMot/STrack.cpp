#include "STrack.h"
std::map<int, int> STrack::_track_count = STrack::InitTrackCount(num_class);

STrack::STrack(cv::Rect_<float>& tlwh, float det_score, cv::Mat temp_feat, int cls_id)
	:_tlwh(tlwh)
{
	_mean.setZero();
	_covariance.setZero();
	_det_score = det_score;
	_class_id = cls_id;
	updateFeatures(temp_feat);
}

STrack::~STrack()
{
}

void STrack::updateFeatures(cv::Mat& feat)
{
	cv::Mat norm_feat;
	cv::normalize(feat, norm_feat, 1.0, 0, cv::NORM_L2);
	feat = norm_feat;
	_curr_feat = feat.clone();
	if (_smooth_feat.empty()) _smooth_feat = feat;
	else
	{
		_smooth_feat = _alpha * _smooth_feat + (1 - _alpha) * feat;
		cv::normalize(_smooth_feat, norm_feat, 1.0, 0, cv::NORM_L2);
		_smooth_feat = norm_feat;
	}
}

void STrack::Predict()
{
	if (_track_state != TrackState::Tracked)
	{
		_mean(7) = 0;
	}
	_kal_filter->predict(_mean, _covariance);
}

// Convert bounding box to format `(center x, center y, aspect ratio,
// height)`, where the aspect ratio is `width / height`.
DETECTBOX STrack::TlwhToXyah(const cv::Rect_<float>& tlwh)
{
	DETECTBOX box;
	float x = tlwh.x + tlwh.width / 2;
	float y = tlwh.y + tlwh.height / 2;
	box << x, y, tlwh.width / tlwh.height, tlwh.height;
	return box;
}

//Start a new tracklet
void STrack::Activate(std::shared_ptr<KalmanFilterTracking> kal_filter, int frame_id)
{
	_kal_filter = kal_filter;
	_track_id = NextTrackID(_class_id);
	auto ret = _kal_filter->initiate(TlwhToXyah(_tlwh));
	_mean = ret.first;
	_covariance = ret.second;
	_tracklet_len = 0;
	_track_state = TrackState::Tracked;
	if (frame_id == 1)
	{
		_is_activated = true;
	}
	_frame_id = frame_id;
	_start_frame = frame_id;
	_trajectory.push_back(_tlwh);
}

void STrack::Reactivate(std::shared_ptr<STrack> new_track, int frame_id, bool new_id)
{
	auto ret = _kal_filter->update(_mean, _covariance, TlwhToXyah(new_track->TlwhToRect()));
	_mean = ret.first;
	_covariance = ret.second;
	_tracklet_len = 0;
	_track_state = TrackState::Tracked;
	_is_activated = true;
	_frame_id = frame_id;
	updateFeatures(new_track->_curr_feat);
	_trajectory.push_back(new_track->TlwhToRect());
	if (new_id)
	{
		_track_id = NextTrackID(_class_id);
	}
}

void STrack::Update(std::shared_ptr<STrack> new_track, int frame_id, bool update_feature)
{
	_frame_id = frame_id;
	_trajectory.push_back(new_track->TlwhToRect());
	_tracklet_len += 1;

	auto ret = _kal_filter->update(_mean, _covariance, TlwhToXyah(new_track->TlwhToRect()));
	_mean = ret.first;
	_covariance = ret.second;

	_track_state = TrackState::Tracked;
	_is_activated = true;
	_det_score = new_track->_det_score;
	if (update_feature)
	{
		updateFeatures(new_track->_curr_feat);
	}
}


DETECTBOX STrack::TlwhToBox()
{
	if (_mean.isZero())
	{
		DETECTBOX box;
		box << _tlwh.x, _tlwh.y, _tlwh.width, _tlwh.height;
		return box;
	}

	DETECTBOX ret = _mean.leftCols(4);
	ret(2) *= ret(3);
	ret.leftCols(2) -= (ret.rightCols(2) / 2);
	return ret;
}

cv::Rect_<float> STrack::TlwhToRect()
{
	DETECTBOX ret = TlwhToBox();
	return cv::Rect_<float>(cv::Point_<float>(ret[0], ret[1]), cv::Point_<float>(ret[0] + ret[2], ret[1] + ret[3]));
}


std::map<int, int> STrack::InitTrackCount(int num_cls)
{
	std::map<int, int>  t_count;
	for (int i = 0; i < num_cls; i++) {
		t_count[i] = 0;
	}
	return t_count;
}
int STrack::NextTrackID(int cls_id)
{
	_track_count[cls_id] += 1;
	return _track_count[cls_id];
}
void STrack::ResetTrackID(int cls_id)
{
	_track_count[cls_id] = 0;
}