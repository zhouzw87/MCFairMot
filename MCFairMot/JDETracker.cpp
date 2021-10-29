#include "JDETracker.h"
#include "Matching.h"


JDETracker::JDETracker(JDETrackerConfig &config, int frame_rate)
	:opt_(config)
{
	frame_id_ = 0;
	det_thresh_ = opt_.conf_thres;
	buffer_size_ = int(frame_rate / 30.0 * opt_.track_buffer);
	max_time_lost_ = opt_.max_lost_time;
	num_class_ = opt_.num_class;
	kal_filter_ = std::shared_ptr<KalmanFilterTracking>(new KalmanFilterTracking());
	for (int i = 0; i < num_class_; i++) {
		track_ids_[i] = 0;
	}
}

JDETracker::~JDETracker()
{
}


std::vector<std::shared_ptr<STrack>> JDETracker::jointStracks(std::vector<std::shared_ptr<STrack>>& tlista, std::vector<std::shared_ptr<STrack>>& tlistb)
{
	std::map<int, int> exists;
	std::vector<std::shared_ptr<STrack>> res;
	for (auto& t: tlista)
	{
		exists[t->GetTrackID()] = 1;
		res.push_back(t);
	}
	for (auto& t : tlistb)
	{
		int tid = t->GetTrackID();
		if (exists.find(tid) == exists.end())
		{
			exists[tid] = 1;
			res.push_back(t);
		}
	}
	return res;
}

std::vector<std::shared_ptr<STrack>> JDETracker::subStracks(std::vector<std::shared_ptr<STrack>>& tlista, std::vector<std::shared_ptr<STrack>>& tlistb)
{
	std::vector<std::shared_ptr<STrack>> res;
	std::map<int, std::shared_ptr<STrack>> stracks;
	for (auto&t:tlista)
	{
		stracks[t->GetTrackID()] = t;
	}
	for (auto&t : tlistb)
	{
		auto tid = t->GetTrackID();
		auto key = stracks.find(tid);
		if (key != stracks.end())
		{
			stracks.erase(key);
		}
	}
	for (auto &v : stracks)
		res.push_back(v.second);
	return res;
}


std::tuple<std::vector<std::shared_ptr<STrack>>, std::vector<std::shared_ptr<STrack>>> JDETracker::removeDuplicateStracks(std::vector<std::shared_ptr<STrack>>& stracksa, std::vector<std::shared_ptr<STrack>>& stracksb)
{
	auto pdist = matching::IouDistance(stracksa, stracksb);
	std::set<int>dupa;
	std::set<int>dupb;
	for (int p = 0; p < pdist.size(); p++)
	{
		for (int q = 0; q < pdist[p].size(); q++)
		{
			if (pdist[p][q] < 0.15f)
			{
				auto timep = stracksa[p]->GetFrameID() - stracksa[p]->GetStartFrame();
				auto timeq = stracksb[q]->GetFrameID() - stracksb[q]->GetStartFrame();
				if (timep > timeq)
					dupb.insert(q);
				else
					dupa.insert(p);
			}
		}
	}
	std::vector<std::shared_ptr<STrack>>resa, resb;
	for (int i = 0; i < stracksa.size(); i++)
	{
		if (dupa.find(i) == dupa.end())
			resa.push_back(stracksa[i]);
	}
	for (int i = 0; i < stracksb.size(); i++)
	{
		if (dupb.find(i) == dupb.end())
			resb.push_back(stracksb[i]);
	}
	return { resa, resb };
}

std::map<int, std::vector<std::shared_ptr<STrack>>> JDETracker::UpdateTracking(std::map<int, std::vector<DetectionBox>>& dets, std::map<int, std::vector<cv::Mat>>& feats)
{
	frame_id_ += 1;
	if (this->frame_id_ % 15000 == 0) {
		for (int cls = 0; cls < num_class_; cls++) {
			track_ids_[cls] = 0;
		}
		frame_id_ = 0;
	}
	std::map<int, std::vector<std::shared_ptr<STrack>>> output_stracks;

	for (int cls = 0; cls < num_class_; cls++) {
		std::vector<DetectionBox> cls_det = dets[cls];
		std::vector<cv::Mat> cls_feature = feats[cls];
		std::vector<std::shared_ptr<STrack>> activated_starcks;
		std::vector<std::shared_ptr<STrack>> refind_stracks;
		std::vector<std::shared_ptr<STrack>> lost_stracks;
		std::vector<std::shared_ptr<STrack>> removed_stracks;
		std::vector<std::shared_ptr<STrack>> detections;
		for (int i = 0; i < cls_det.size(); i++)
			detections.push_back(std::shared_ptr<STrack>(new STrack(cls_det[i].box, cls_det[i].score, cls_feature[i], cls)));

		//Add newly detected tracklets to tracked_stracks'''
		std::vector<std::shared_ptr<STrack>> unconfirmed_stracks;
		std::vector<std::shared_ptr<STrack>> tracked_stracks;
		for (auto& track : tracked_stracks_[cls]) {
			if (!track->IsActivated()) {
				unconfirmed_stracks.push_back(track);
			}
			else {
				tracked_stracks.push_back(track);
			}
		}
		/******Step 2: First association, with embedding*/
		auto strack_pool = jointStracks(tracked_stracks, lost_stracks_[cls]);
		/**********Predict the current location with KF*/
		for (auto& strack : strack_pool)
			strack->Predict();
		/******for strack in strack_pool */
		auto dists = matching::EmbeddingDistance(strack_pool, detections);
		matching::FuseMotion(kal_filter_, dists, strack_pool, detections);
		//auto[matches, u_track, u_detection] = matching::linear_assignment(dists, 0.4f, strack_pool.size(), detections.size());
		std::tuple<std::vector<cv::Point>, std::set<int>, std::set<int>> tuple_out;
		tuple_out = matching::LinearAssignment(dists, 0.4f, strack_pool.size(), detections.size());
		std::vector<cv::Point> matches = std::get<0>(tuple_out);
		std::set<int> u_track = std::get<1>(tuple_out);
		std::set<int> u_detection = std::get<2>(tuple_out);
		for (auto pt : matches)
		{
			auto& track = strack_pool[pt.x];
			auto& det = detections[pt.y];
			if (track->GetTrackState() == TrackState::Tracked)
			{
				track->Update(det, frame_id_);
				activated_starcks.push_back(track);
			}
			else
			{
				track->Reactivate(det, frame_id_, false);
				refind_stracks.push_back(track);
			}
		}
		/********''' Step 3: Second association, with IOU'''*/
		std::vector<std::shared_ptr<STrack>> detections_tmp;
		for (auto& ud : u_detection) detections_tmp.push_back(detections[ud]);
		detections = detections_tmp;

		std::vector<std::shared_ptr<STrack>> r_tracked_stracks;
		for (auto& ut : u_track)
		{
			if (strack_pool[ut]->GetTrackState() == TrackState::Tracked)
			{
				r_tracked_stracks.push_back(strack_pool[ut]);
			}
		}
		dists = matching::IouDistance(r_tracked_stracks, detections);
		//auto[matches2, u_track2, u_detection2] = matching::linear_assignment(dists, 0.5f, r_tracked_stracks.size(), detections.size());
		std::tuple<std::vector<cv::Point>, std::set<int>, std::set<int>> tuple_out2;
		tuple_out2 = matching::LinearAssignment(dists, 0.5f, r_tracked_stracks.size(), detections.size());
		std::vector<cv::Point> matches2 = std::get<0>(tuple_out2);
		std::set<int> u_track2 = std::get<1>(tuple_out2);
		std::set<int> u_detection2 = std::get<2>(tuple_out2);
		for (auto pt : matches2)
		{
			auto& track = r_tracked_stracks[pt.x];
			auto& det = detections[pt.y];
			if (track->GetTrackState() == TrackState::Tracked)
			{
				track->Update(det, frame_id_);
				activated_starcks.push_back(track);
			}
			else
			{
				track->Reactivate(det, frame_id_, false);
				refind_stracks.push_back(track);
			}
		}
		for (auto& it : u_track2)
		{
			auto& track = r_tracked_stracks[it];
			if (track->GetTrackState() != TrackState::Lost)
			{
				track->MarkLost();
				lost_stracks.push_back(track);
			}
		}
		/*****Deal with unconfirmed tracks, usually tracks with only one beginning frame'''*/
		detections_tmp.clear();
		for (auto& ud : u_detection2) detections_tmp.push_back(detections[ud]);
		detections = detections_tmp;
		dists = matching::IouDistance(unconfirmed_stracks, detections);
		//auto[matches3, u_unconfirmed3, u_detection3] = matching::linear_assignment(dists, 0.7f, unconfirmed.size(), detections.size());
		std::tuple<std::vector<cv::Point>, std::set<int>, std::set<int>> tuple_out3;
		tuple_out3 = matching::LinearAssignment(dists, 0.7f, unconfirmed_stracks.size(), detections.size());
		std::vector<cv::Point> matches3 = std::get<0>(tuple_out3);
		std::set<int> u_unconfirmed3 = std::get<1>(tuple_out3);
		std::set<int> u_detection3 = std::get<2>(tuple_out3);
		for (auto pt : matches3)
		{
			unconfirmed_stracks[pt.x]->Update(detections[pt.y], frame_id_);
			activated_starcks.push_back(unconfirmed_stracks[pt.x]);
		}
		for (auto& it : u_unconfirmed3)
		{
			auto& track = unconfirmed_stracks[it];
			track->MarkRemove();
			removed_stracks.push_back(track);
		}
		/**********Step 4: Init new stracks"""*/
		for (auto& inew : u_detection3)
		{
			auto& track = detections[inew];
			if (track->GetDetScore() < det_thresh_)
			{
				continue;
			}
			track_ids_[cls]++;
			track->Activate(kal_filter_, frame_id_, track_ids_[cls]);
			activated_starcks.push_back(track);
		}
		/**********Step 5: Update state"""*/
		for (auto& track : lost_stracks_[cls])
		{
			if ((frame_id_ - track->GetFrameID()) > max_time_lost_)
			{
				track->MarkRemove();
				removed_stracks.push_back(track);
			}
		}
		std::vector<std::shared_ptr<STrack>> tracked_stracks_tmp;
		for (auto& t : tracked_stracks_[cls])
		{
			if (t->GetTrackState() == TrackState::Tracked)
			{
				tracked_stracks_tmp.push_back(t);
			}
		}
		tracked_stracks_[cls] = tracked_stracks_tmp;
		tracked_stracks_[cls] = jointStracks(tracked_stracks_[cls], activated_starcks);
		tracked_stracks_[cls] = jointStracks(tracked_stracks_[cls], refind_stracks);
		lost_stracks_[cls] = subStracks(lost_stracks_[cls], tracked_stracks_[cls]);
		lost_stracks_[cls].insert(lost_stracks_[cls].end(), lost_stracks.begin(), lost_stracks.end());
		lost_stracks_[cls] = subStracks(lost_stracks_[cls], removed_stracks_[cls]);
		removed_stracks_[cls].insert(removed_stracks_[cls].end(), removed_stracks.begin(), removed_stracks.end());
		std::tuple<std::vector<std::shared_ptr<STrack>>, std::vector<std::shared_ptr<STrack>>> tuple_strack;
		tuple_strack = removeDuplicateStracks(tracked_stracks_[cls], lost_stracks_[cls]);
		std::vector<std::shared_ptr<STrack>> stracksa = std::get<0>(tuple_strack);
		std::vector<std::shared_ptr<STrack>> stracksb = std::get<1>(tuple_strack);
		//auto[stracksa, stracksb] = remove_duplicate_stracks(this->tracked_stracks, this->lost_stracks);
		tracked_stracks_[cls] = stracksa;
		lost_stracks_[cls] = stracksb;
		for (auto& track : tracked_stracks_[cls])
		{
			if (track->IsActivated())
			{
				output_stracks[cls].push_back(track);
			}
		}

	}
	return output_stracks;
}
