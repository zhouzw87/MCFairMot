#include "Matching.h"
#include "Hungarian.h"
namespace matching
{
	float iouValue(const cv::Rect_<float>& bb_det, const cv::Rect_<float>& bb_pre)
	{
		float in = (bb_det & bb_pre).area();
		float un = bb_det.area() + bb_pre.area() - in;

		if (un < DBL_EPSILON)
			return 0;

		return (float)(in / un);
	}

	std::vector<std::vector<double>> IouDistance(std::vector<std::shared_ptr<STrack>>& atracks, std::vector<std::shared_ptr<STrack>>& btracks)
	{
		std::vector<std::vector<double>> cost_matrix;
		cost_matrix.resize(atracks.size(), std::vector<double>(btracks.size(), 0));
		if (cost_matrix.empty()) return cost_matrix;
		for (int i = 0; i < atracks.size(); i++)
		{
			for (int j = 0; j < btracks.size(); j++)
			{
				float dist = iouValue(atracks[i]->TlwhToRect(), btracks[j]->TlwhToRect());
				cost_matrix[i][j] = 1 - dist;
			}
		}
		return cost_matrix;
	}

	std::vector<std::vector<double>> EmbeddingDistance(std::vector<std::shared_ptr<STrack>>&tracks, std::vector<std::shared_ptr<STrack>>&detections, std::string metric)
	{
		std::vector<std::vector<double>> cost_matrix;
		cost_matrix.resize(tracks.size(), std::vector<double>(detections.size(), 0));
		if (cost_matrix.empty()) return cost_matrix;
		for (int i = 0; i < tracks.size(); i++)
		{
			for (int j = 0; j < detections.size(); j++)
			{
				float dist = tracks[i]->GetSmoothFeat().dot(detections[j]->GetCurrentFeat());
				cost_matrix[i][j] = 1 - dist;
			}
		}
		return cost_matrix;
	}

	std::tuple<std::vector<cv::Point>, std::set<int>, std::set<int>> LinearAssignment(std::vector<std::vector<double>>& dists, float thresh, int dim_a, int dim_b)
	{
		std::vector<cv::Point> matchedPairs;
		std::vector<float> matchedScores;
		std::set<int> unmatchedTrajectories;
		std::set<int> unmatchedDetections;
		std::vector<int> assignment;
		std::set<int> allItems;
		std::set<int> matchedItems;
		if (dists.empty())
		{
			for (int i = 0; i < dim_a; i++)unmatchedTrajectories.insert(i);
			for (int i = 0; i < dim_b; i++)unmatchedDetections.insert(i);
			return { matchedPairs , unmatchedTrajectories , unmatchedDetections };
		}
		unsigned int trkNum = 0;
		unsigned int detNum = 0;
		///////////////////////////////////////
		// 3.2. associate detections to tracked object (both represented as bounding boxes)
		trkNum = dists.size();
		detNum = (trkNum > 0) ? (dists[0].size()) : 0;

		// solve the assignment problem using hungarian algorithm.
		// the resulting assignment is [track(prediction) : detection], with len=preNum
		HungarianAlgorithm HungAlgo;
		assignment.clear();
		HungAlgo.Solve(dists, assignment);

		// find matches, unmatched_detections and unmatched_predictions
		unmatchedTrajectories.clear();
		unmatchedDetections.clear();
		allItems.clear();
		matchedItems.clear();

		if (detNum > trkNum) //	there are unmatched detections
		{
			for (unsigned int n = 0; n < detNum; n++)
				allItems.insert(n);

			for (unsigned int i = 0; i < trkNum; ++i)
				matchedItems.insert(assignment[i]);

			set_difference(allItems.begin(), allItems.end(),
				matchedItems.begin(), matchedItems.end(),
				std::insert_iterator<std::set<int>>(unmatchedDetections, unmatchedDetections.begin()));
		}
		else if (detNum < trkNum) // there are unmatched trajectory/predictions
		{
			for (unsigned int i = 0; i < trkNum; ++i)
				if (assignment[i] == -1) // unassigned label will be set as -1 in the assignment algorithm
					unmatchedTrajectories.insert(i);
		}
		else
			;

		// filter out matched with low IOU
		matchedPairs.clear();
		for (unsigned int i = 0; i < trkNum; ++i)
		{
			if (assignment[i] == -1) // pass over invalid values
				continue;
			if (1 - dists[i][assignment[i]] < thresh)
			{
				unmatchedTrajectories.insert(i);
				unmatchedDetections.insert(assignment[i]);
			}
			else
			{
				matchedPairs.push_back(cv::Point(i, assignment[i]));
				matchedScores.push_back(1 - dists[i][assignment[i]]);
			}
		}
		return { matchedPairs , unmatchedTrajectories , unmatchedDetections };
	}


	void FuseMotion(std::shared_ptr<KalmanFilterTracking>kf
		, std::vector<std::vector<double>>&cost_matrix
		, std::vector<std::shared_ptr<STrack>>&tracks
		, std::vector<std::shared_ptr<STrack>>&detections
		, bool only_position
		, float lambda_)
	{
		if (cost_matrix.empty()) return;
		int gating_dim = (only_position == true ? 2 : 4);
		double gating_threshold = KalmanFilterTracking::chi2inv95[gating_dim];
		std::vector<DETECTBOX> measurements;
		for (auto&det : detections)
		{
			auto box = det->TlwhToXyah(det->TlwhToRect());
			measurements.push_back(box);
		}
		for (int i = 0; i < tracks.size(); i++)
		{
			auto&track = tracks[i];
			Eigen::Matrix<float, 1, -1> gating_distance = kf->gating_distance(
				track->GetMean(), track->GetCovariance(), measurements, only_position);
			for (int j = 0; j < gating_distance.cols(); j++)
			{
				if (gating_distance(0, j) > gating_threshold)  cost_matrix[i][j] = INFTY_COST;
				else cost_matrix[i][j] = lambda_ * cost_matrix[i][j] + (1 - lambda_) * gating_distance[j];
			}
		}
	}
}