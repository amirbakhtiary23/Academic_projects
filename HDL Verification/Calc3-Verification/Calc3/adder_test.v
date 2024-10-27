module adder_test;

   wire [0:63] bin_sum;
   wire        bin_ovfl;
   reg [0:63]  fxu_areg_q, fxu_breg_q;
   reg [0:3]   alu_cmd;

   adder(bin_ovfl, bin_sum, alu_cmd, fxu_areg_q, fxu_breg_q);

   initial
     begin
	$display ("time: %t,alu_cmd: %d,reg1: %d, reg2: %d\nresult: %d, ovfl: %b", $time, alu_cmd, fxu_areg_q, fxu_breg_q, bin_sum, bin_ovfl);
	 fxu_areg_q = 25;
	fxu_breg_q = 22;
	alu_cmd = 1;

	#5
	$display ("time: %t,alu_cmd: %d,reg1: %d, reg2: %d\nresult: %d, ovfl: %b", $time, alu_cmd, fxu_areg_q, fxu_breg_q, bin_sum, bin_ovfl);
	  alu_cmd = 2;
	

	#5
	  $display ("time: %t,alu_cmd: %d,reg1: %d, reg2: %d\nresult: %d, ovfl: %b", $time, alu_cmd, fxu_areg_q, fxu_breg_q, bin_sum, bin_ovfl);
	  fxu_areg_q = 10;
	fxu_breg_q = 10;

	#5
	  $display ("time: %t,alu_cmd: %d,reg1: %d, reg2: %d\nresult: %d, ovfl: %b", $time, alu_cmd, fxu_areg_q, fxu_breg_q, bin_sum, bin_ovfl);
	  alu_cmd = 1;

	#5
	  $display ("time: %t,alu_cmd: %d,reg1: %d, reg2: %d\nresult: %d, ovfl: %b", $time, alu_cmd, fxu_areg_q, fxu_breg_q, bin_sum, bin_ovfl);
	  fxu_areg_q = 3'b100;
	fxu_breg_q = 2'b10;


	#5 
	  $display ("time: %t,alu_cmd: %d,reg1: %d, reg2: %d\nresult: %d, ovfl: %b", $time, alu_cmd, fxu_areg_q, fxu_breg_q, bin_sum, bin_ovfl);
	  alu_cmd = 2;

	#5
	  $display ("time: %t,alu_cmd: %d,reg1: %d, reg2: %d\nresult: %d, ovfl: %b", $time, alu_cmd, fxu_areg_q, fxu_breg_q, bin_sum, bin_ovfl);
	  fxu_areg_q = 0;
	fxu_breg_q =0;

	#5
	  $display ("time: %t,alu_cmd: %d,reg1: %d, reg2: %d\nresult: %d, ovfl: %b", $time, alu_cmd, fxu_areg_q, fxu_breg_q, bin_sum, bin_ovfl);
	fxu_areg_q=2;
	fxu_breg_q=4;
	#5
	  $display ("time: %t,alu_cmd: %d,reg1: %d, reg2: %d\nresult: %d, ovfl: %b", $time, alu_cmd, fxu_areg_q, fxu_breg_q, bin_sum, bin_ovfl);
	

	#100	
	$stop;

     end // initial begin


endmodule