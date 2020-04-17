//----------------------------------------------------------
// Copyright 2019 University of Oxford
// Written by Michael A. Boemo (mb915@cam.ac.uk)
// This software is licensed under GPL-3.0.  You should have
// received a copy of the license with this software.  If
// not, please Email the author.
//----------------------------------------------------------

#include <algorithm>
#include <exception>
#include <vector>
#include "poreModels.h"
#include "error_handling.h"
#define _USE_MATH_DEFINES


inline double uniformPDF( double lb, double ub, double x ){

	if ( x >= lb && x <= ub ){

		return 1.0/( ub - lb );
	}
	else {
		return 0.0;
	}
};


inline double normalPDF( double mu, double sigma, double x ){

	return ( 1.0/sqrt( 2.0*pow( sigma, 2.0 )*M_PI ) )*exp( -pow( x - mu , 2.0 )/( 2.0*pow( sigma, 2.0 ) ) );
}
