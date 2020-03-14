#include "Timer.h"

void Timer::start(std::string name) {
	TimePoint tp;
	ch::high_resolution_clock::time_point start =
	    ch::high_resolution_clock::now();
	tp.start = start;
	this->tpDict[name] = tp;
}

double Timer::end(std::string name, bool shouldLog) {
	if (this->tpDict.find(name) == this->tpDict.end()) {
		printf("key %s not found!\n", name.c_str());
		return 0;
	}
	TimePoint tp = this->tpDict[name];
	ch::high_resolution_clock::time_point end =
	    ch::high_resolution_clock::now();
	tp.end = end;
	tp.duration =
	    ch::duration_cast<ch::duration<double>>(tp.end - tp.start).count();
	this->tpDict[name] = tp;
	if (shouldLog) {
		printf("%s time: %g\n", name.c_str(), tp.duration);
	}
	return tp.duration;
}

double Timer::duration(std::string name) {
	if (this->tpDict.find(name) == this->tpDict.end()) {
		printf("key %s not found!\n", name.c_str());
		return 0;
	}
	TimePoint tp = tpDict[name];
	if (tp.end.time_since_epoch().count()) {
		return this->tpDict[name].duration;
	}
	ch::high_resolution_clock::time_point now =
	    ch::high_resolution_clock::now();
	return ch::duration_cast<ch::duration<double>>(now - tp.start).count();
}
