//
// Created by smallflyfly on 2021/7/20.
//

#include "Tracker.h"

Tracker::Tracker() {

}

Tracker::Tracker(NearestNeighborDistanceMetric metric, float maxIouDistance, int maxAge, int nInit) {
    this->metric = metric;
    this->maxIouDistance = maxIouDistance;
    this->maxAge = maxAge;
    this->nInit = nInit;
    this->kf = KalmanFilter();
}

void Tracker::predict() {
    for (int i = 0; i < this->tracks.size(); ++i) {
        Track track = this->tracks[i];
        track.
    }
}
