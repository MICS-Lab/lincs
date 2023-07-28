#ifndef COMMUNICATIONLIST_LQSF093AEJL__H
#define COMMUNICATIONLIST_LQSF093AEJL__H

#include <list>
#include <vector>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <optional>
#include <cassert>
#include <iostream>


namespace MaLib {



template <class T>
class CommunicationList {
    std::mutex _mutex;
    std::condition_variable _cv_push;
    std::condition_variable _cv_wait;

    std::list<T> data;
    unsigned int numberWaiting=0;

    bool _closed = false;
public:

    CommunicationList() {

    }

    unsigned int size() {
        std::scoped_lock<std::mutex> lock(_mutex);
        return data.size();
    }

    void clear() {
        std::scoped_lock<std::mutex> lock(_mutex);
        assert(numberWaiting == 0);
        _closed = false;
        //newWaintingProcess = false;
        numberWaiting=0;
        data.clear();
    }

    /*
     * No more elements will be added.
     * Processes waiting in pop() will be unlocked.
     */
    void close() {
        {
            std::scoped_lock<std::mutex> lock(_mutex);
            _closed = true;
        }
        _cv_push.notify_all();
    }

    void push(const T& element) {
        {
            std::scoped_lock<std::mutex> lock(_mutex);
            assert(!_closed);
            data.push_back(element);
        }
        _cv_push.notify_one();
    }

    template <class T2>
    void pushAll(const T2 &elements) {
        {
            std::scoped_lock<std::mutex> lock(_mutex);
            assert(!_closed);
            data.insert(data.end(), elements.begin(), elements.end());
        }
        _cv_push.notify_all();
    }

    const std::list<T> & getWithoutRemove_unsafe() {
        return data;
    }


    unsigned int getNumberWaiting() {
        std::lock_guard<std::mutex> lock(_mutex);
        return numberWaiting;
    }


    /*
     * Blocks until a new element is added or the method close() is called.
     * return nothing if the CommunicationList is empty and closed.
     */
    std::optional<T> pop() {
        std::unique_lock<std::mutex> lock(_mutex);

        if((data.size() == 0) && (_closed == false)) {
            ++numberWaiting;
            //newWaintingProcess = true;
            _cv_wait.notify_all();

            _cv_push.wait(lock, [&]{
                return (data.size() || _closed);
            });

            assert(data.size() || _closed);
            assert(numberWaiting > 0);
            --numberWaiting;
        }

        if(data.size()) {
            auto result = data.front();
            data.pop_front();
            return result;
        }
        assert(_closed);

        return {};
    }

    /*
     * Wait until at least $n$ processes are wainting in pop().
     */
    unsigned int areWaiting(unsigned int n) {
        std::unique_lock<std::mutex> lock(_mutex);
        assert(!_closed);


        if( std::max(0, static_cast<int>(numberWaiting) - static_cast<int>(data.size()) ) >= n ) {
            return static_cast<int>(numberWaiting) - static_cast<int>(data.size());
        }

        _cv_wait.wait(lock, [&]{
            return std::max(0, static_cast<int>(numberWaiting) - static_cast<int>(data.size())) >= n;
        });

        assert( std::max(0, static_cast<int>(numberWaiting) - static_cast<int>(data.size())) >= n );
        return static_cast<int>(numberWaiting) - static_cast<int>(data.size());
    }

};

}

#endif // COMMUNICATIONLIST_LQSF093AEJL__H

