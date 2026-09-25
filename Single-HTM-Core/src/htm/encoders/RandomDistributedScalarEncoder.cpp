/* ---------------------------------------------------------------------
 * HTM Community Edition of NuPIC
 * Copyright (C) 2019, David McDougall
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero Public License version 3 as
 * published by the Free Software Foundation.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 * See the GNU Affero Public License for more details.
 *
 * You should have received a copy of the GNU Affero Public License
 * along with this program.  If not, see http://www.gnu.org/licenses.
 * --------------------------------------------------------------------- */

/** @file
 * Implementation of the RandomDistributedScalarEncoder
 */

#include <htm/encoders/RandomDistributedScalarEncoder.hpp>
#include <murmurhash3/MurmurHash3.hpp>
#include <htm/utils/Random.hpp>
#include <algorithm> // fill

using namespace std;
using namespace htm;

RandomDistributedScalarEncoder::RandomDistributedScalarEncoder(
                                              const RDSE_Parameters &parameters)
  { initialize( parameters ); }

void RandomDistributedScalarEncoder::initialize( const RDSE_Parameters &parameters)
{
  // Check size parameter
  NTA_CHECK( parameters.size > 0u );

  // Initialize parent class.
  BaseEncoder<Real64>::initialize({ parameters.size });

  // Check other parameters
  UInt num_active_args = 0;
  if( parameters.activeBits > 0u)   { num_active_args++; }
  if( parameters.sparsity   > 0.0f) { num_active_args++; }
  NTA_CHECK( num_active_args != 0u )
      << "Missing argument, need one of: 'activeBits' or 'sparsity'.";
  NTA_CHECK( num_active_args == 1u )
      << "Too many arguments, choose only one of: 'activeBits' or 'sparsity'.";

  UInt num_resolution_args = 0;
  if( parameters.radius     > 0.0f) { num_resolution_args++; }
  if( parameters.category )         { num_resolution_args++; }
  if( parameters.resolution > 0.0f) { num_resolution_args++; }
  NTA_CHECK( num_resolution_args != 0u )
      << "Missing argument, need one of: 'radius', 'resolution', 'category'.";
  NTA_CHECK( num_resolution_args == 1u )
      << "Too many arguments, choose only one of: 'radius', 'resolution', 'category'.";

  args_ = parameters;
  // Finish filling in all of parameters.

  // Determine number of activeBits.
  if( args_.sparsity > 0.0f ) {
    NTA_CHECK( args_.sparsity >= 0.0f );
    NTA_CHECK( args_.sparsity <= 1.0f );
    args_.activeBits = (UInt) round( args_.size * args_.sparsity );
    NTA_CHECK( args_.activeBits > 0u );
  }
  // Determine sparsity. Always calculate this even if it was given, to correct for rounding error.
  args_.sparsity = (Real) args_.activeBits / args_.size;

  if( args_.category ) {
    args_.radius = 1.0f;
  }

  // Determine resolution.
  if( args_.radius > 0.0f ) {
    args_.resolution = args_.radius / args_.activeBits;
  }
  // Determine radius.
  else if( args_.resolution > 0.0f ) {
    args_.radius = args_.activeBits * args_.resolution;
  }

  while( args_.seed == 0u ) {
    args_.seed = Random().getUInt32();
  }

  NTA_CHECK(check_parameters()) << "Failed hash collision resistance check, please increase size, sparsity, and or activeBits.";
}

void RandomDistributedScalarEncoder::encode(Real64 input, SDR &output)
{
  // Check inputs
  NTA_CHECK( output.size == size );
  if( isnan(input) ) {
    output.zero();
    return;
  }
  else if( args_.category ) {
    NTA_CHECK( input == Real64(UInt64(input)))
      << "Input to category encoder must be an unsigned integer!";
  }

  auto &data = output.getDense();
  fill( data.begin(), data.end(), 0u );

  // Bucket index of the input.
  //
  // The index used to be `(UInt) (input / args_.resolution)`. That cast is
  // undefined in C++ whenever the quotient is negative or does not fit in a
  // UInt, and its result then depends on the build. An x86-64 baseline/AVX2
  // build truncates toward zero into 64 bits and keeps the low 32 bits: a
  // quotient between 2^32 and 2^63 lands on an arbitrary bucket, and one at
  // or beyond 2^63 (e.g. a tiny resolution) on bucket 0, the exact code of 0.
  // An AVX-512 build returns 0xFFFFFFFF for every negative or too-large
  // quotient, and that code is almost the code of bucket 0.
  //
  // The computation below is defined for every input:
  //   * inside +/- kMaxBucket it gives exactly the bucket the x86-64
  //     baseline/AVX2 build gave before (truncate toward zero in 64 bits,
  //     keep the low 32 bits), so encodings of in-range inputs do not change;
  //   * beyond it, the quotient is clamped to the last bucket on its own side
  //     of the scale, so an out-of-range input takes the most extreme
  //     encoding the feature has: the top bucket for a positive input, the
  //     bottom bucket for a negative one, and never a code near bucket 0.
  // kMaxBucket is 2^30, so the top and the bottom buckets sit 2^31 apart in
  // the UInt ring and their activeBits-long windows never overlap.
  constexpr Real64 kMaxBucket = 1073741824.0;   // 2^30
  Real64 quotient = input / args_.resolution;
  if( quotient > kMaxBucket ) {
    quotient = kMaxBucket;
  }
  else if( quotient < -kMaxBucket ) {
    quotient = -kMaxBucket;
  }
  const Int64 signed_index = static_cast<Int64>( quotient );   // truncates toward zero
  const UInt  index = static_cast<UInt>( static_cast<UInt64>( signed_index ) );

  for(auto offset = 0u; offset < args_.activeBits; ++offset)
  {
    UInt hash_buffer = index + offset;
    UInt32 bucket = MurmurHash3_x86_32(&hash_buffer, sizeof(hash_buffer), args_.seed);
    bucket = bucket % size;

    // Don't worry about hash collisions.  Instead measure the critical
    // properties of the encoder in unit tests and quantify how significant
    // the hash collisions are.  This encoder can not fix the collisions
    // because it does not record past encodings.  Collisions cause small
    // deviations in the sparsity or semantic similarity, depending on how
    // they're handled.

    data[bucket] = 1u;
  }
  output.setDense( data );
}

bool RandomDistributedScalarEncoder::check_parameters() {
  if( parameters.size >= 1000 && parameters.activeBits >= 10 ) {
    return true;
  }
  const auto maximum_overlap = 0.33f;
  const auto number_of_trials = 1000u;
  SDR a({parameters.size});
  SDR b({parameters.size});
  // BUGFIX (determinism): upstream called the single-argument randomize(),
  // which seeds from the hardware entropy source -- so for borderline
  // parameters this acceptance check was a per-RUN coin flip: the very same
  // (size, sparsity, seed) configuration could construct fine today and
  // throw tomorrow. The trial RNG is now derived from the encoder's own
  // seed, making construction a pure function of its parameters. Encoding
  // output was never affected (it is MurmurHash-based); only the
  // accept/reject behaviour of this check is made reproducible.
  Random trial_rng( args_.seed );
  a.randomize(parameters.sparsity, trial_rng);
  for(auto i = 0u; i < number_of_trials; i++) {
    b.randomize(parameters.sparsity, trial_rng);
    Real overlap = Real(a.getOverlap(b)) / parameters.activeBits;
    if( overlap > maximum_overlap ) {
      return false;
    }
  }
  return true;
}

std::ostream & htm::operator<<(std::ostream & out, const RandomDistributedScalarEncoder &self)
{
  out << "RDSE ";
  out << "  size:       " << self.parameters.size << ",\n";
  out << "  activeBits: " << self.parameters.activeBits << ",\n";
  out << "  resolution: " << self.parameters.resolution << ",\n";
  out << "  category:   " << self.parameters.category << ",\n";
  out << "  seed:       " << self.parameters.seed << std::endl;
  return out;
}