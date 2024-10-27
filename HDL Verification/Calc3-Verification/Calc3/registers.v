//  Library:  calc3
//  Module:  Registers
//  Author:  Naseer Siddique

module registers (adder_read_d1, adder_read_d2, shift_read_d1, shift_read_d2, a_clk, b_clk, c_clk, adder_read_adr1, adder_read_adr2, adder_read_valid1, adder_read_valid2, adder_write_adr, adder_write_data, adder_write_valid, reset, shift_read_adr1, shift_read_adr2, shift_read_valid1, shift_read_valid2, shift_write_adr, shift_write_data, shift_write_valid);
   output [0:63] adder_read_d1, adder_read_d2, shift_read_d1, shift_read_d2;
   
   input 	 a_clk, b_clk, c_clk, adder_read_valid1, adder_read_valid2, adder_write_valid, reset, shift_read_valid1, shift_read_valid2, shift_write_valid;
   input [0:3] 	 adder_read_adr1, adder_read_adr2, adder_write_adr, shift_read_adr1, shift_read_adr2, shift_write_adr;
   input [0:31]  adder_write_data, shift_write_data;
   
   reg [0:31] 	 reg0, reg1, reg2, reg3, reg4, reg5, reg6, reg7, reg8, reg9, reg10, reg11, reg12, reg13, reg14, reg15;
   wire [0:31] 	 reg_val;
   
   assign 	 adder_read_d1[0:31] = 32'b0;
   assign 	 adder_read_d2[0:31] = 32'b0;
   assign 	 shift_read_d1[0:31] = 32'b0;
   assign 	 shift_read_d2[0:31] = 32'b0;
   assign 	 reg_val[0:31] = { {12{1'b1}}, 1'b0, {19{1'b1}} };
   
   assign 	 adder_read_d1[32:63] = 
		 (~adder_read_valid1) ? 32'b0 :
		 (adder_read_adr1 == 4'b0000) ? reg0 :
		 (adder_read_adr1 == 4'b0001) ? reg1 :
		 (adder_read_adr1 == 4'b0010) ? reg2 :
		 (adder_read_adr1 == 4'b0011) ? reg3 :
		 (adder_read_adr1 == 4'b0100) ? reg4 :
		 (adder_read_adr1 == 4'b0101) ? reg5 :
		 (adder_read_adr1 == 4'b0110) ? reg6 :
		 (adder_read_adr1 == 4'b0111) ? reg7 :
		 (adder_read_adr1 == 4'b1000) ? reg8 :
		 (adder_read_adr1 == 4'b1001) ? reg9 :
		 (adder_read_adr1 == 4'b1010) ? reg10 :
		 (adder_read_adr1 == 4'b1011) ? reg11 :
		 (adder_read_adr1 == 4'b1100) ? reg12 :
		 (adder_read_adr1 == 4'b1101) ? reg13 :
		 (adder_read_adr1 == 4'b1110) ? reg14 :
		 reg15;
   
   assign 	 adder_read_d2[32:63] = 
		 (~adder_read_valid2) ? 32'b0 :
		 (adder_read_adr2 == 4'b0000) ? reg0 :
		 (adder_read_adr2 == 4'b0001) ? reg1 :
		 (adder_read_adr2 == 4'b0010) ? reg2 :
		 (adder_read_adr2 == 4'b0011) ? reg3 :
		 (adder_read_adr2 == 4'b0100) ? reg4 :
		 (adder_read_adr2 == 4'b0101) ? reg5 :
		 (adder_read_adr2 == 4'b0110) ? reg6 :
		 (adder_read_adr2 == 4'b0111) ? reg7 :
		 (adder_read_adr2 == 4'b1000) ? reg8 :
		 (adder_read_adr2 == 4'b1001) ? reg9 :
		 (adder_read_adr2 == 4'b1010) ? reg10 :
		 (adder_read_adr2 == 4'b1011) ? reg11 :
		 (adder_read_adr2 == 4'b1100) ? reg12 :
		 (adder_read_adr2 == 4'b1101) ? reg13 & reg_val :
		 (adder_read_adr2 == 4'b1110) ? reg14 :
		 reg15;
   
   
   assign 	 shift_read_d1[32:63] = 
		 (~shift_read_valid1) ? 32'b0 :
		 (shift_read_adr1 == 4'b0000) ? reg0 :
		 (shift_read_adr1 == 4'b0001) ? reg1 :
		 (shift_read_adr1 == 4'b0010) ? reg2 :
		 (shift_read_adr1 == 4'b0011) ? reg3 :
		 (shift_read_adr1 == 4'b0100) ? reg4 :
		 (shift_read_adr1 == 4'b0101) ? reg5 :
		 (shift_read_adr1 == 4'b0110) ? reg6 :
		 (shift_read_adr1 == 4'b0111) ? reg7 :
		 (shift_read_adr1 == 4'b1000) ? reg8 :	
		 (shift_read_adr1 == 4'b1001) ? reg9 :
		 (shift_read_adr1 == 4'b1010) ? reg10 :
		 (shift_read_adr1 == 4'b1011) ? reg11 :
		 (shift_read_adr1 == 4'b1100) ? reg12 :
		 (shift_read_adr1 == 4'b1101) ? reg13 :
		 (shift_read_adr1 == 4'b1110) ? reg14 :
		 reg15;

   assign 	 shift_read_d2[32:63] = 
		 (~shift_read_valid2) ? 32'b0 :
		 (shift_read_adr2 == 4'b0000) ? reg0 :
		 (shift_read_adr2 == 4'b0001) ? reg1 :
		 (shift_read_adr2 == 4'b0010) ? reg2 :
		 (shift_read_adr2 == 4'b0011) ? reg3 :
		 (shift_read_adr2 == 4'b0100) ? reg4 :
		 (shift_read_adr2 == 4'b0101) ? reg5 :
		 (shift_read_adr2 == 4'b0110) ? reg6 :
		 (shift_read_adr2 == 4'b0111) ? reg7 :
		 (shift_read_adr2 == 4'b1000) ? reg8 :
		 (shift_read_adr2 == 4'b1001) ? reg9 :
		 (shift_read_adr2 == 4'b1010) ? reg10 :
		 (shift_read_adr2 == 4'b1011) ? reg11 :
		 (shift_read_adr2 == 4'b1100) ? reg12 :
		 (shift_read_adr2 == 4'b1101) ? reg13 :
		 (shift_read_adr2 == 4'b1110) ? reg14 :
		 reg15;

   always
     @ (negedge c_clk) begin
	
	reg0 <= (reset) ? 32'b0 :
		((0 == (adder_write_adr)) && adder_write_valid) ? adder_write_data :
		((0 == (shift_write_adr)) && shift_write_valid) ? shift_write_data :
		reg0;
	
     end
   
   always
     @ (negedge c_clk) begin
	
	reg1 <= (reset) ? 32'b0 :
		((1 == (adder_write_adr)) && adder_write_valid) ? adder_write_data :
		((1 == (shift_write_adr)) && shift_write_valid) ? shift_write_data :
		reg1;
	
     end
   
   always
     @ (negedge c_clk) begin
	
	reg2 <= (reset) ? 32'b0 :
		((2 == (adder_write_adr)) && adder_write_valid) ? adder_write_data :
		((2 == (shift_write_adr)) && shift_write_valid) ? shift_write_data :
		reg2;
	
     end
   
   always
     @ (negedge c_clk) begin
	
	reg3 <= (reset) ? 32'b0 :
		((3 == (adder_write_adr)) && adder_write_valid) ? adder_write_data :
		((3 == (shift_write_adr)) && shift_write_valid) ? shift_write_data :
		reg3;
	
     end
   
   always
     @ (negedge c_clk) begin
	
	reg4 <= (reset) ? 32'b0 :
		((4 == (adder_write_adr)) && adder_write_valid) ? adder_write_data :
		((4 == (shift_write_adr)) && shift_write_valid) ? shift_write_data :
		reg4;
	
     end

   always
     @ (negedge c_clk) begin
	
	reg5 <= (reset) ? 32'b0 :
		((5 == (adder_write_adr)) && adder_write_valid) ? adder_write_data :
		((5 == (shift_write_adr)) && shift_write_valid) ? shift_write_data :
		reg5;
	
     end
   
   always
     @ (negedge c_clk) begin

	reg6 <= (reset) ? 32'b0 :
		((6 == (adder_write_adr)) && adder_write_valid) ? adder_write_data :
		((6 == (shift_write_adr)) && shift_write_valid) ? shift_write_data :
		reg6;
	
     end

   always
     @ (negedge c_clk) begin
	
	reg7 <= (reset) ? 32'b0 :
		((7 == (adder_write_adr)) && adder_write_valid) ? adder_write_data :
		((7 == (shift_write_adr)) && shift_write_valid) ? shift_write_data :
		reg7;
	
     end
   
   always
     @ (negedge c_clk) begin
	
	reg8 <= (reset) ? 32'b0 :
		((8 == (adder_write_adr)) && adder_write_valid) ? adder_write_data :
		((8 == (shift_write_adr)) && shift_write_valid) ? shift_write_data :
		reg8;
	
     end
   
   always
     @ (negedge c_clk) begin

	reg9 <= (reset) ? 32'b0 :
		((9 == (adder_write_adr)) && adder_write_valid) ? adder_write_data :
		((9 == (shift_write_adr)) && shift_write_valid) ? shift_write_data :
		reg9;

     end

   always
     @ (negedge c_clk) begin

	reg10 <= (reset) ? 32'b0 :
		 ((10 == (adder_write_adr)) && adder_write_valid) ? adder_write_data :
		 ((10 == (shift_write_adr)) && shift_write_valid) ? shift_write_data :
		 reg10;

     end

   always
     @ (negedge c_clk) begin

	reg11 <= (reset) ? 32'b0 :
		 ((11 == (adder_write_adr)) && adder_write_valid) ? adder_write_data :
		 ((11 == (shift_write_adr)) && shift_write_valid) ? shift_write_data :
		 reg11;

     end

   always
     @ (negedge c_clk) begin

	reg12 <= (reset) ? 32'b0 :
		 ((12 == (adder_write_adr)) && adder_write_valid) ? adder_write_data :
		 ((12 == (shift_write_adr)) && shift_write_valid) ? shift_write_data :
		 reg12;

     end
   
   always
     @ (negedge c_clk) begin

	reg13 <= (reset) ? 32'b0 :
		 ((13 == (adder_write_adr)) && adder_write_valid) ? adder_write_data :
		 ((13 == (shift_write_adr)) && shift_write_valid) ? shift_write_data :
		 reg13;

     end

   always
     @ (negedge c_clk) begin

	reg14 <= (reset) ? 32'b0 :
		 ((14 == (adder_write_adr)) && adder_write_valid) ? adder_write_data :
		 ((14 == (shift_write_adr)) && shift_write_valid) ? shift_write_data :
		 reg14;

     end

   always
     @ (negedge c_clk) begin
	
	reg15 <= (reset) ? 32'b0 :
		 ((15 == (adder_write_adr)) && adder_write_valid) ? adder_write_data :
		 ((15 == (shift_write_adr)) && shift_write_valid) ? shift_write_data :
		 reg15;
	
     end

endmodule