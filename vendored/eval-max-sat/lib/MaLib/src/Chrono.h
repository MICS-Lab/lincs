#ifndef CHRONO_R7U52KTM

#define CHRONO_R7U52KTM

#include <chrono>
#include <cstdio>
#include <iostream>


namespace MaLib
{
    class Chrono
    {
        public :

            /* Removed for lincs
            Chrono(std::string name, bool afficherQuandDetruit=true)
                : _name(name), _duree(0),_dureeSec(0), _pause(false), _afficherQuandDetruit(afficherQuandDetruit)
            {
                gettimeofday(&depart, &tz);
            }
            */  // Removed for lincs

            Chrono()
                : _duree(0),_dureeSec(0), _pause(false)
            {
                depart = std::chrono::steady_clock::now();
            }

            /* Removed for lincs
            ~Chrono() {
                if(_name.size())
                    if(_afficherQuandDetruit)
                        print();
            }

            void setDuree(long sec, long microSec=0)
            {
                _duree= sec * 1000000L + microSec;
                _dureeSec=sec;
            }
            */  // Removed for lincs

            void tic()
            {
                _pause=false;
                _duree=0;
                _dureeSec=0;
                depart = std::chrono::steady_clock::now();
            }

            long pause(bool val)
            {
                if(val)
                {
                    if(!_pause)
                    {
                        fin = std::chrono::steady_clock::now();
                        _duree += std::chrono::duration_cast<std::chrono::microseconds>(fin - depart).count();
                        _dureeSec += std::chrono::duration_cast<std::chrono::seconds>(fin - depart).count();
                        _pause=true;
                    }
                }else
                {
                    if(_pause)
                    {
                        depart = std::chrono::steady_clock::now();
                        _pause=false;
                    }
                }
                return _duree;
            }

            /* Removed for lincs
            long pauseSec(bool val)
            {
                if(val)
                {
                    if(!_pause)
                    {
                        gettimeofday(&fin, &tz);
                        _duree += (fin.tv_sec-depart.tv_sec) * 1000000L + (fin.tv_usec-depart.tv_usec);
                        _dureeSec += fin.tv_sec-depart.tv_sec ;
                        _pause=true;
                    }
                }else
                {
                    if(_pause)
                    {
                        gettimeofday(&depart, &tz);
                        _pause=false;
                    }
                }
                return _dureeSec;
            }
            */  // Removed for lincs

            long tac()
            {
                if(_pause==false)
                {
                    fin = std::chrono::steady_clock::now();
                    return std::chrono::duration_cast<std::chrono::microseconds>(fin - depart).count() + _duree;
                }else
                {
                    return _duree;
                }
            }

	    long tacSec()
            {
                if(_pause==false)
                {
                    fin = std::chrono::steady_clock::now();
                    return std::chrono::duration_cast<std::chrono::seconds>(fin - depart).count() + _dureeSec;
                }else
                {
                    return _dureeSec;
                }
            }

            /* Removed for lincs
            void print()
            {
                double val = tac();
                if(_name.size())
                    std::cout << _name << ": ";
                if(val < 1000.0)
                    std::cout << val << " µs" << std::endl;
                else if(val < 1000000.0)
                    std::cout << val/1000.0 << " ms" << std::endl;
                else
                    std::cout << val/1000000.0 << " sec" << std::endl;
                //static double total; total+=val/1000000.0; std::cout<<"Total : "<<total<<" s"<<std::endl; // TODO: Remove this line
            }

            void afficherQuandDetruit(bool val) {
                _afficherQuandDetruit = val;
            }
            */  // Removed for lincs

        private :

        /* Removed for lincs
        std::string _name;
        */  // Removed for lincs
        std::chrono::steady_clock::time_point depart, fin;
        long _duree;
        long _dureeSec;

        bool _pause;
        /* Removed for lincs
        bool _afficherQuandDetruit;
        */  // Removed for lincs
    };
}


#endif /* end of include guard: CHRONO_R7U52KTM */




