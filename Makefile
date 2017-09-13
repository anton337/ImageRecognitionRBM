all:
	g++ -O3 -g0 -o digits main.cpp readBMP.c -lboost_thread -lboost_system -lGL -lglut -lGLU -lfftw3 -lfftw3f;
	#g++ -O0 -g3 -o digits main.cpp readBMP.c -lboost_thread -lboost_system -lGL -lglut -lGLU -lfftw3 -lfftw3f;
	./digits
