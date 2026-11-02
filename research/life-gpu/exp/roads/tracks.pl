#!/usr/bin/env perl
# Reads a saved world: how the track field (fc.y) and the food piles are distributed.
# Usage: tracks.pl world.sav [tracks.pgm]
use strict; use warnings;
my ($path, $pgm) = @ARGV;
open(my $f, '<:raw', $path) or die "$path: $!";
read($f, my $n, 8); my $len = unpack('Q<', $n);
read($f, my $h, $len);
my ($size) = $h =~ /"size":(\d+)/;
my ($tick) = $h =~ /"tick":(\d+)/;
my $sq = $size * $size;
# File order: occ seen seen_next terrain fa fb fc ...
my $base = 8 + $len;
seek($f, $base, 0); read($f, my $occ, $sq * 4);
seek($f, $base + $sq * 12, 0); read($f, my $ter, $sq * 4);
seek($f, $base + $sq * 16, 0); read($f, my $fa, $sq * 16);
seek($f, $base + $sq * 32, 0); read($f, my $fb, $sq * 16);
seek($f, $base + $sq * 48, 0); read($f, my $fc, $sq * 16);
my (@y, $pile, $rich, $open, @sc);
my $img = '';
for my $i (0 .. $sq - 1) {
    my $t = unpack('L<', substr($ter, $i * 4, 4));
    my $v = unpack('f<', substr($fc, $i * 16 + 4, 4));
    if ($pgm) { $img .= chr(($t >> 24) & 1 ? 40 : int(255 * (($v / 2 > 1 ? 1 : $v / 2) ** 0.5))); }
    next if ($t >> 24) & 1;
    $open++;
    push @y, $v;
    my $g = unpack('f<', substr($fa, $i * 16, 4));
    if ($g > 4) { $rich++; $pile += $g - 4; }
    my @s = unpack('f<4', substr($fb, $i * 16, 16));
    $sc[$_] += $s[$_] for 0 .. 3;
}
@y = sort { $a <=> $b } @y;
my $sum = 0; $sum += $_ for @y;
my $pct = sub { $y[int($#y * $_[0])] };
my $top = 0; $top += $y[$_] for int($#y * 0.95) .. $#y;
my $top1 = 0; $top1 += $y[$_] for int($#y * 0.99) .. $#y;
printf "tick %d mean %.3f p50 %.3f p90 %.3f p99 %.3f top5%%share %.3f top1%%share %.3f rich_squares %d (%.2f%%) pile %.0f scent/sq %.3f %.3f %.3f %.3f\n",
    $tick, $sum / @y, $pct->(.5), $pct->(.9), $pct->(.99), $top / ($sum || 1), $top1 / ($sum || 1),
    $rich // 0, 100 * ($rich // 0) / $open, $pile // 0, map { ($_ // 0) / $open } @sc[0 .. 3];
if ($pgm) { open(my $o, '>:raw', $pgm) or die; print $o "P5\n$size $size\n255\n", $img; }
