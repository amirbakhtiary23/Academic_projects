`ifndef GUARD_GENERATOR
`define GUARD_GENERATOR
`include "Driver.sv"
`include "Interface.sv"
class rNumber; //a class for creating random numbers
randc logic [0:31] data;
randc logic [0:31] data2;
randc logic [0:31] data3;
randc logic [0:31] data4;
randc logic [0:3] d1;
randc logic [0:3] d2;
randc logic [0:3] r1;
randc logic [0:3] r3;
randc logic [0:3] r2;
randc logic [0:3] r4;
randc logic [0:3] d3;
randc logic [0:3] d4;
randc logic [0:3] temp;
randc logic [0:1] tag;
randc logic [0:1] tag2;
randc logic [0:1] tag3;
randc logic [0:1] tag4;
randc logic [0:31] overflowDetector1;
randc logic [0:31] overflowDetector2;
function void random();
data=$urandom_range (32'd0,32'd4294967295);
data3=$urandom_range (32'd0,32'd4294967295);
data4=$urandom_range (32'd0,32'd4294967295);
data2=$urandom_range (32'd0,32'd15);
d1=$urandom_range(0,15);
d2=$urandom_range(0,15);
d4=$urandom_range(0,15);
d3=$urandom_range(0,15);
r1=$urandom_range(0,15);
r2=$urandom_range(0,15);
r3=$urandom_range(0,15);
r4=$urandom_range(0,15);
tag= $urandom_range (0,3);
tag2= $urandom_range (0,3);
tag3= $urandom_range (0,3);
tag4= $urandom_range (0,3);
temp=$urandom_range (0,3);
endfunction
function void randomOverflowDetector();
overflowDetector1=$urandom_range (32'd2147483648,32'd4294967295);
overflowDetector2=$urandom_range (32'd2147483648,32'd4294967295);
endfunction
endclass


//endclass

class Generator;

virtual GlobalPorts globalPorts;
virtual Stimuli port1;
virtual Stimuli port2;
virtual Stimuli port3;
virtual Stimuli port4;
Driver driver1;
Driver driver2;
Driver driver3;
Driver driver4;
rNumber randomNumber;
function new (virtual GlobalPorts globalPorts,
virtual Stimuli _port1, virtual Stimuli _port2,
virtual Stimuli _port3, virtual Stimuli _port4);
begin
this.port1=_port1;
this.port2=_port2;
this.port3=_port3;
this.port4=_port4;

this.globalPorts = globalPorts;
driver1=new(1,globalPorts,port1);
driver2=new(2,globalPorts,port2);
driver3=new(3,globalPorts,port3);
driver4=new(4,globalPorts,port4);
end
endfunction

task start();
begin
$display ("Generator started");
this.runTests();
$display ("Generator finished");
end
endtask

task runTests();
begin
this.reset();
this.tc_write1();
this.tc_write2();
this.tc_write3();
this.tc_write4();
this.tc_overflow1();
this.tc_overflow2();
this.tc_overflow3();
this.tc_overflow4();
this.tc_overflow1_();
this.tc_overflow2_();
this.tc_overflow3_();
this.tc_overflow4_();
this.tc_shiftL1();
this.tc_shiftL2();
this.tc_shiftL3();
this.tc_shiftL4();
this.tc_shiftR1();
this.tc_shiftR2();
this.tc_shiftR3();
this.tc_shiftR4();
this.tc_cc1();
this.tc_cc2();
this.reset();
end
endtask

task tc_write1();
begin
for (int i = 0 ; i<16;i++)
begin//repeat (100) begin 

this.randomNumber=new();
this.randomNumber.random();

@ (posedge globalPorts.clock);
$display("Tag is ",this.randomNumber.tag);
driver1.store(this.randomNumber.data,i,this.randomNumber.tag);
$display("d1 is %0b",i);
$display("data is %0b",this.randomNumber.data);
@(posedge globalPorts.clock);
$display("Tag is ",this.randomNumber.tag);
driver1.fetch(i,this.randomNumber.tag);

$display ("Fetching %0b",i);
end
end
endtask


task tc_write2();
begin
for (int i = 0 ; i<16;i++)
repeat (100) begin 

this.randomNumber=new();
this.randomNumber.random();

@ (posedge globalPorts.clock);
driver2.store(this.randomNumber.data,i,this.randomNumber.tag);

$display("d1 is %0b",i);
$display("data is %0b",this.randomNumber.data);
@(posedge globalPorts.clock);
driver2.fetch(i,this.randomNumber.tag);
$display ("Fetching %0b",i);
end
end
endtask

task tc_write3();
begin
for (int i = 0 ; i<16;i++)
repeat (100) begin 

this.randomNumber=new();
this.randomNumber.random();

@ (posedge globalPorts.clock);
driver3.store(this.randomNumber.data,i,this.randomNumber.tag);
$display("d1 is %0b",i);
$display("data is %0b",this.randomNumber.data);
@(posedge globalPorts.clock);
driver3.fetch(i,this.randomNumber.tag);
$display ("Fetching %0b",i);
end
end
endtask


task tc_write4();
begin
for (int i = 0 ; i<16;i++)
repeat (100) begin 

this.randomNumber=new();
this.randomNumber.random();

@ (posedge globalPorts.clock);
driver4.store(this.randomNumber.data,i,this.randomNumber.tag);
$display("d1 is %0b",i);
$display("data is %0b",this.randomNumber.data);
@(posedge globalPorts.clock);
driver4.fetch(i,this.randomNumber.tag);
$display ("Fetching %0b",i);
end
end
endtask



task tc_overflow1();
begin
for (int i=0; i<16;i++) begin
for (int j=0; j<16 ; j++) begin
if (j==i)
continue;
repeat (5) begin
this.randomNumber=new();
this.randomNumber.random();
this.randomNumber.randomOverflowDetector();

@ (posedge globalPorts.clock);

driver1.store(this.randomNumber.overflowDetector1,i,this.randomNumber.tag);

@ (posedge globalPorts.clock);

driver1.store(this.randomNumber.overflowDetector2,j,this.randomNumber.tag2);

@ (posedge globalPorts.clock);

driver1.add(i,j,this.randomNumber.r1,this.randomNumber.tag3);

@ (posedge globalPorts.clock);

driver1.fetch(this.randomNumber.r1,this.randomNumber.tag4);
end
end 
end
end
endtask
task tc_overflow2();
begin
for (int i=0; i<16;i++) begin
for (int j=0; j<16 ; j++) begin
if (j==i)
continue;
repeat (5) begin
this.randomNumber=new();
this.randomNumber.random();
this.randomNumber.randomOverflowDetector();

@ (posedge globalPorts.clock);

driver2.store(this.randomNumber.overflowDetector1,i,this.randomNumber.tag);

@ (posedge globalPorts.clock);

driver2.store(this.randomNumber.overflowDetector2,j,this.randomNumber.tag2);

@ (posedge globalPorts.clock);

driver2.add(i,j,this.randomNumber.r1,this.randomNumber.tag3);

@ (posedge globalPorts.clock);

driver2.fetch(this.randomNumber.r1,this.randomNumber.tag4);
end
end 
end
end
endtask
task tc_overflow3();
begin
for (int i=0; i<16;i++) begin
for (int j=0; j<16 ; j++) begin
if (j==i)
continue;
repeat (5) begin
this.randomNumber=new();
this.randomNumber.random();
this.randomNumber.randomOverflowDetector();

@ (posedge globalPorts.clock);

driver3.store(this.randomNumber.overflowDetector1,i,this.randomNumber.tag);

@ (posedge globalPorts.clock);

driver3.store(this.randomNumber.overflowDetector2,j,this.randomNumber.tag2);

@ (posedge globalPorts.clock);

driver3.add(i,j,this.randomNumber.r1,this.randomNumber.tag3);

@ (posedge globalPorts.clock);

driver3.fetch(this.randomNumber.r1,this.randomNumber.tag4);
end
end 
end
end
endtask

task tc_overflow4();
begin
for (int i=0; i<16;i++) begin
for (int j=0; j<16 ; j++) begin
if (j==i)
continue;
repeat (5) begin
this.randomNumber=new();
this.randomNumber.random();
this.randomNumber.randomOverflowDetector();

@ (posedge globalPorts.clock);

driver4.store(this.randomNumber.overflowDetector1,i,this.randomNumber.tag);

@ (posedge globalPorts.clock);

driver4.store(this.randomNumber.overflowDetector2,j,this.randomNumber.tag2);

@ (posedge globalPorts.clock);

driver4.add(i,j,this.randomNumber.r1,this.randomNumber.tag3);

@ (posedge globalPorts.clock);

driver4.fetch(this.randomNumber.r1,this.randomNumber.tag4);
end
end 
end
end
endtask





task reset();
begin
@(posedge globalPorts.clock)
this.port1.reqx_cmd=0;
this.port1.reqx_d1=0;
this.port1.reqx_d2=0;
this.port1.reqx_r1=0;
this.port1.reqx_tag=0;
this.port1.reqx_data=0;

this.port2.reqx_cmd=0;
this.port2.reqx_d1=0;
this.port2.reqx_d2=0;
this.port2.reqx_r1=0;
this.port2.reqx_tag=0;
this.port2.reqx_data=0;

this.port3.reqx_cmd=0;
this.port3.reqx_d1=0;
this.port3.reqx_d2=0;
this.port3.reqx_r1=0;
this.port3.reqx_tag=0;
this.port3.reqx_data=0;

this.port4.reqx_cmd=0;
this.port4.reqx_d1=0;
this.port4.reqx_d2=0;
this.port4.reqx_r1=0;
this.port4.reqx_tag=0;
this.port4.reqx_data=0;
globalPorts.reset=1;
repeat (5) @ (posedge globalPorts.clock);
globalPorts.reset=0;
$display ("reset done");
end
endtask

task tc_overflow1_();
begin
for (int i=0; i<16;i++) begin
for (int j=0; j<16 ; j++) begin
if (j==i)
continue;
repeat (5) begin
this.randomNumber=new();
this.randomNumber.random();
this.randomNumber.randomOverflowDetector();

@ (posedge globalPorts.clock);

driver1.store(this.randomNumber.overflowDetector1,i,this.randomNumber.tag);

@ (posedge globalPorts.clock);

driver1.store(this.randomNumber.overflowDetector2,j,this.randomNumber.tag2);

@ (posedge globalPorts.clock);

driver1.sub(i,j,this.randomNumber.r1,this.randomNumber.tag3);

@ (posedge globalPorts.clock);

driver1.fetch(this.randomNumber.r1,this.randomNumber.tag4);
end
end 
end
end
endtask

task tc_overflow2_();
begin
for (int i=0; i<16;i++) begin
for (int j=0; j<16 ; j++) begin
if (j==i)
continue;
repeat (5) begin
this.randomNumber=new();
this.randomNumber.random();
this.randomNumber.randomOverflowDetector();

@ (posedge globalPorts.clock);

driver2.store(this.randomNumber.overflowDetector1,i,this.randomNumber.tag);

@ (posedge globalPorts.clock);

driver2.store(this.randomNumber.overflowDetector2,j,this.randomNumber.tag2);

@ (posedge globalPorts.clock);

driver2.sub(i,j,this.randomNumber.r1,this.randomNumber.tag3);

@ (posedge globalPorts.clock);

driver2.fetch(this.randomNumber.r1,this.randomNumber.tag4);
end
end 
end
end
endtask

task tc_overflow3_();
begin
for (int i=0; i<16;i++) begin
for (int j=0; j<16 ; j++) begin
if (j==i)
continue;
repeat (5) begin
this.randomNumber=new();
this.randomNumber.random();
this.randomNumber.randomOverflowDetector();

@ (posedge globalPorts.clock);

driver3.store(this.randomNumber.overflowDetector1,i,this.randomNumber.tag);

@ (posedge globalPorts.clock);

driver3.store(this.randomNumber.overflowDetector2,j,this.randomNumber.tag2);

@ (posedge globalPorts.clock);

driver3.sub(i,j,this.randomNumber.r1,this.randomNumber.tag3);

@ (posedge globalPorts.clock);

driver3.fetch(this.randomNumber.r1,this.randomNumber.tag4);
end
end 
end
end
endtask


task tc_overflow4_();
begin
for (int i=0; i<16;i++) begin
for (int j=0; j<16 ; j++) begin
if (j==i)
continue;
repeat (5) begin
this.randomNumber=new();
this.randomNumber.random();
this.randomNumber.randomOverflowDetector();

@ (posedge globalPorts.clock);

driver4.store(this.randomNumber.overflowDetector1,i,this.randomNumber.tag);

@ (posedge globalPorts.clock);

driver4.store(this.randomNumber.overflowDetector2,j,this.randomNumber.tag2);

@ (posedge globalPorts.clock);

driver4.sub(i,j,this.randomNumber.r1,this.randomNumber.tag3);

@ (posedge globalPorts.clock);

driver4.fetch(this.randomNumber.r1,this.randomNumber.tag4);
end
end 
end
end
endtask


task tc_shiftL4();
begin
for (int i=0; i<16;i++) begin
for (int j=0; j<16 ; j++) begin
if (j==i)
continue;
repeat (5) begin
this.randomNumber=new();
this.randomNumber.random();
this.randomNumber.randomOverflowDetector();

@ (posedge globalPorts.clock);

driver4.store(this.randomNumber.data,i,this.randomNumber.tag);

@ (posedge globalPorts.clock);

driver4.store(this.randomNumber.data2,j,this.randomNumber.tag2);

@ (posedge globalPorts.clock);

driver4.shiftLeft(i,j,this.randomNumber.r1,this.randomNumber.tag3);

@ (posedge globalPorts.clock);

driver4.fetch(this.randomNumber.r1,this.randomNumber.tag4);
end
end 
end
end
endtask

task tc_shiftL3();
begin
for (int i=0; i<16;i++) begin
for (int j=0; j<16 ; j++) begin
if (j==i)
continue;
repeat (5) begin
this.randomNumber=new();
this.randomNumber.random();
this.randomNumber.randomOverflowDetector();

@ (posedge globalPorts.clock);

driver3.store(this.randomNumber.data,i,this.randomNumber.tag);

@ (posedge globalPorts.clock);

driver3.store(this.randomNumber.data2,j,this.randomNumber.tag2);

@ (posedge globalPorts.clock);

driver3.shiftLeft(i,j,this.randomNumber.r1,this.randomNumber.tag3);

@ (posedge globalPorts.clock);

driver3.fetch(this.randomNumber.r1,this.randomNumber.tag4);
end
end 
end
end
endtask


task tc_shiftL2();
begin
for (int i=0; i<16;i++) begin
for (int j=0; j<16 ; j++) begin
if (j==i)
continue;
repeat (5) begin
this.randomNumber=new();
this.randomNumber.random();
this.randomNumber.randomOverflowDetector();

@ (posedge globalPorts.clock);

driver2.store(this.randomNumber.data,i,this.randomNumber.tag);

@ (posedge globalPorts.clock);

driver2.store(this.randomNumber.data2,j,this.randomNumber.tag2);

@ (posedge globalPorts.clock);

driver2.shiftLeft(i,j,this.randomNumber.r1,this.randomNumber.tag3);

@ (posedge globalPorts.clock);

driver2.fetch(this.randomNumber.r1,this.randomNumber.tag4);
end
end 
end
end
endtask


task tc_shiftL1();
begin
for (int i=0; i<16;i++) begin
for (int j=0; j<16 ; j++) begin
if (j==i)
continue;
repeat (5) begin
this.randomNumber=new();
this.randomNumber.random();
this.randomNumber.randomOverflowDetector();

@ (posedge globalPorts.clock);

driver1.store(this.randomNumber.data,i,this.randomNumber.tag);

@ (posedge globalPorts.clock);

driver1.store(this.randomNumber.data2,j,this.randomNumber.tag2);

@ (posedge globalPorts.clock);

driver1.shiftLeft(i,j,this.randomNumber.r1,this.randomNumber.tag3);

@ (posedge globalPorts.clock);

driver1.fetch(this.randomNumber.r1,this.randomNumber.tag4);
end
end 
end
end
endtask
task tc_shiftR4();
begin
for (int i=0; i<16;i++) begin
for (int j=0; j<16 ; j++) begin
if (j==i)
continue;
repeat (5) begin
this.randomNumber=new();
this.randomNumber.random();
this.randomNumber.randomOverflowDetector();

@ (posedge globalPorts.clock);

driver4.store(this.randomNumber.data,i,this.randomNumber.tag);

@ (posedge globalPorts.clock);

driver4.store(this.randomNumber.data2,j,this.randomNumber.tag2);

@ (posedge globalPorts.clock);

driver4.shiftRight(i,j,this.randomNumber.r1,this.randomNumber.tag3);

@ (posedge globalPorts.clock);

driver4.fetch(this.randomNumber.r1,this.randomNumber.tag4);
end
end 
end
end
endtask

task tc_shiftR2();
begin
for (int i=0; i<16;i++) begin
for (int j=0; j<16 ; j++) begin
if (j==i)
continue;
repeat (5) begin
this.randomNumber=new();
this.randomNumber.random();
this.randomNumber.randomOverflowDetector();

@ (posedge globalPorts.clock);

driver2.store(this.randomNumber.data,i,this.randomNumber.tag);

@ (posedge globalPorts.clock);

driver2.store(this.randomNumber.data2,j,this.randomNumber.tag2);

@ (posedge globalPorts.clock);

driver2.shiftRight(i,j,this.randomNumber.r1,this.randomNumber.tag3);

@ (posedge globalPorts.clock);

driver2.fetch(this.randomNumber.r1,this.randomNumber.tag4);
end
end 
end
end
endtask
task tc_shiftR3();
begin
for (int i=0; i<16;i++) begin
for (int j=0; j<16 ; j++) begin
if (j==i)
continue;
repeat (5) begin
this.randomNumber=new();
this.randomNumber.random();
this.randomNumber.randomOverflowDetector();

@ (posedge globalPorts.clock);

driver3.store(this.randomNumber.data,i,this.randomNumber.tag);

@ (posedge globalPorts.clock);

driver3.store(this.randomNumber.data2,j,this.randomNumber.tag2);

@ (posedge globalPorts.clock);

driver3.shiftRight(i,j,this.randomNumber.r1,this.randomNumber.tag3);

@ (posedge globalPorts.clock);

driver3.fetch(this.randomNumber.r1,this.randomNumber.tag4);
end
end 
end
end
endtask



task tc_shiftR1();
begin
for (int i=0; i<16;i++) begin
for (int j=0; j<16 ; j++) begin
if (j==i)
continue;
repeat (5) begin
this.randomNumber=new();
this.randomNumber.random();
this.randomNumber.randomOverflowDetector();

@ (posedge globalPorts.clock);

driver1.store(this.randomNumber.data,i,this.randomNumber.tag);

@ (posedge globalPorts.clock);

driver1.store(this.randomNumber.data2,j,this.randomNumber.tag2);

@ (posedge globalPorts.clock);

driver1.shiftRight(i,j,this.randomNumber.r1,this.randomNumber.tag3);

@ (posedge globalPorts.clock);

driver1.fetch(this.randomNumber.r1,this.randomNumber.tag4);
end
end 
end
end
endtask

task tc_cc1();
repeat (5) begin
this.randomNumber=new();
this.randomNumber.random();

@ (posedge globalPorts.clock);

driver1.store(this.randomNumber.data,this.randomNumber.d1,this.randomNumber.tag);
driver2.store(this.randomNumber.data3,this.randomNumber.temp,this.randomNumber.tag2);
driver3.store(this.randomNumber.data2,this.randomNumber.d2,this.randomNumber.tag3);
@ (posedge globalPorts.clock);

driver2.add(this.randomNumber.d1,this.randomNumber.d2,this.randomNumber.r1,this.randomNumber.tag4);
driver1.add(this.randomNumber.d1,this.randomNumber.temp,this.randomNumber.r1,this.randomNumber.tag2);

@ (posedge globalPorts.clock);

driver1.fetch(this.randomNumber.r1,this.randomNumber.tag4);
driver2.fetch(this.randomNumber.r1,this.randomNumber.tag3);

end

endtask

task tc_cc2();
repeat (15) begin
this.randomNumber=new();
this.randomNumber.random();

@ (posedge globalPorts.clock);

driver1.store(this.randomNumber.data,this.randomNumber.d1,this.randomNumber.tag);
driver2.store(this.randomNumber.data3,this.randomNumber.temp,this.randomNumber.tag2);
driver3.store(this.randomNumber.data2,this.randomNumber.d2,this.randomNumber.tag3);
driver4.store(this.randomNumber.data4,this.randomNumber.d4,this.randomNumber.tag4);
@ (posedge globalPorts.clock);

driver2.add(this.randomNumber.d1,this.randomNumber.d2,this.randomNumber.r2,this.randomNumber.tag4);
driver1.add(this.randomNumber.d3,this.randomNumber.temp,this.randomNumber.r2,this.randomNumber.tag2);
driver4.add(this.randomNumber.d2,this.randomNumber.temp,this.randomNumber.r3,this.randomNumber.tag3);
driver3.add(this.randomNumber.d1,this.randomNumber.d3,this.randomNumber.r4,this.randomNumber.tag);
driver3.shiftLeft(this.randomNumber.d1,this.randomNumber.temp,this.randomNumber.r1,this.randomNumber.tag);
@ (posedge globalPorts.clock);
driver4.fetch(this.randomNumber.d1,this.randomNumber.tag2);
end


endtask

endclass
`endif