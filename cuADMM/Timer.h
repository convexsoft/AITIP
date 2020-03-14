#include <chrono>
#include <map>

#ifndef TIMER_H
#define TIMER_H

namespace ch = std::chrono;

typedef struct {
	ch::high_resolution_clock::time_point start;
	ch::high_resolution_clock::time_point end;
	double duration;
} TimePoint;

class Timer {
	std::map<std::string, TimePoint> tpDict;

   public:
	void start(std::string name);
	double end(std::string name, bool shouldLog = true);
	double duration(std::string name);
};

#endif
