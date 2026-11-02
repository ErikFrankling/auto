#!/usr/bin/env perl
# Read saved physical hardness and correlate it with measured diets and movement.
use strict;
use warnings;
use JSON::PP;
open my $f, '<:raw', $ARGV[0] or die $!;
read($f, my $len, 8)==8 or die 'header';
read($f, my $json, unpack('Q<',$len));
my $head=decode_json($json);
my $start=8+length($json)+64*$head->{size}**2;
seek($f,$start,0);
my @cells;
for (1..$head->{slots}) { read($f,my $b,64)==64 or die 'cell'; push @cells,[unpack('f<16',$b)]; }
my @groups=map {{n=>0,hardness=>0,moving=>0,bite=>0,ground=>0,x0_budget=>0}} 0..2;
for my $c (@cells) {
 read($f,my $b,112)==112 or die 'stat';
 my @s=unpack('f<28',$b);
 next unless $c->[0]>0;
 my $h=$c->[4]; my $g=$groups[$h<.2 ? 0 : $h<.6 ? 1 : 2];
 $g->{n}++; $g->{hardness}+=$h; $g->{moving}+=$s[19]; $g->{x0_budget}+=$s[14];
 my $total=$s[0]+$s[1]+$s[2]+$s[3];
 if($total>0) { $g->{ground}+=$s[0]/$total; $g->{bite}+=$s[2]/$total; }
}
for my $g (@groups) { for my $k (qw(hardness moving bite ground x0_budget)) { $g->{$k}/=$g->{n} if $g->{n}; } }
 my @sketch_groups = ([],[],[]);
for my $c (@cells) {
 read($f,my $b,16)==16 or die q(sketch);
 next unless $c->[0]>0;
 my $bin=$c->[4]<.2 ? 0 : $c->[4]<.6 ? 1 : 2;
 push @{$sketch_groups[$bin]}, [unpack(q(c16),$b)] if @{$sketch_groups[$bin]}<128;
}
sub gap { my ($a,$b)=@_; my $sum=0; for(0..15) { $sum+=($a->[$_]-$b->[$_])**2; } return sqrt($sum/16)*4/127; }
my %gaps;
for my $pair ([0,0],[2,2],[0,2]) {
 my ($a,$b)=@$pair; my ($sum,$n)=(0,0);
 for my $i (0..$#{$sketch_groups[$a]}) {
 for my $j (0..$#{$sketch_groups[$b]}) {
 next if $a==$b && $i>=$j;
 $sum+=gap($sketch_groups[$a][$i],$sketch_groups[$b][$j]); $n++;
 }}
 $gaps{qq($a-$b)}={pairs=>$n,mean_gap=>$n ? $sum/$n : undef};
}
print JSON::PP->new->canonical->pretty->encode({tick=>$head->{tick},bins=>\@groups,sketch_gaps=>\%gaps});
