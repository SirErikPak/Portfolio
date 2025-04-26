if exists (select * from dbo.sysobjects where id = object_id(N'[dbo].[P_GECS_Address_Fix]') and OBJECTPROPERTY(id, N'IsProcedure') = 1)
drop procedure [dbo].[P_GECS_Address_Fix]
GO
Create Proc P_GECS_Address_Fix
As
/*********************************************************************/
/*       WYETH Address –Re-Format Address at GECS                    */
/*********************************************************************/
/*     This is a BASIC Process to Re-Format Address (1-3)            */
/*  This Process will re-distributed 100 characters address fields   */
/*               into 60 character address fields                    */
/*                                                                   */
/*                                                                   */
/*                                                                   */
/*                ---Date---  ---name---  --Purpose----------------- */
/*  Created       09/14/2009  EPAK                                   */
/*  Mod 01          /  /       .                                     */
/*********************************************************************/
SET NOCOUNT ON	

/*********************************************************************/
/*		Declare Internal Variables			     */
/*********************************************************************/
DECLARE @Address1	Varchar(60),
	@Address2	Varchar(60),
	@Address3	Varchar(40),
	@TAddress	Varchar(300),
	@GID		Varchar(12),
	@OPT_ID		Varchar(21),
	@Max		Int,
	@SeqNo		Int,
	@iSeqNo		Int,
	@First		Int,
	@AddressCounter	Int,
	@X		Char(1)

/*********************************************************************/
/*                                                                   */
/*********************************************************************/
Create	Table	#Address
 (
	SeqNo			Int		Identity,
	GID			Varchar(12)	NOT NULL,
	OPT_ID			Varchar(21)	NOT NULL,
	Address1		Varchar(100)	NULL,
	Address2		Varchar(100)	NULL,
	Address3		Varchar(100)	NULL
 )

/*********************************************************************/
/*                                                                   */
/*********************************************************************/
Insert	#Address
 (
	GID,OPT_ID,Address1,Address2,Address3
 )
Select	GLOBAL_ID,OPT_ID,RTRIM(Address1),RTRIM(Address2),RTRIM(Address3)
From	tblGECSemployee
Where	LEN(Address1) > 60
Or	LEN(Address2) > 60
Or	LEN(Address3) > 60

/*********************************************************************/
/*                                                                   */
/*********************************************************************/
SET	@SeqNo	= 1

Select	@Max	=	Count(SeqNo)
From	#Address

/*********************************************************************/
/*								     */
/*********************************************************************/
WHILE @SeqNo <= @Max
 Begin	---- BEGIN OUTER LOOP
	/*****************************************************************/
	/*						                 */
	/*****************************************************************/
	Select	@TAddress	= RTRIM(ISNULL(Address1,'')) + ' ' +  RTRIM(ISNULL(Address2,'')) + ' ' + RTRIM(ISNULL(Address3,'')),
		@GID		= GID,
		@OPT_ID 	= OPT_ID
	From	#Address
	Where	SeqNo		= @SeqNo

	/*****************************************************************/
	/*						                 */
	/*****************************************************************/
	SET 	@iSeqNo 	= LEN(@TAddress)
	SET	@AddressCounter = 0
	SET	@First		= 60

	/*****************************************************************/
	/*						                 */
	/*****************************************************************/
	WHILE @iSeqNo >= 1
	 Begin	---- Begin INNER LOOP

		/*********************************************************/
		/*					                 */
		/*********************************************************/
		Select	@X	=	SUBSTRING(@TAddress,@First,1)


		IF @X = SPACE(1) AND @AddressCounter = 0
		 Begin

			/*************************************************/
			/*				                 */
			/*************************************************/
			Select	@Address1 = SUBSTRING(@TAddress,1,@First)

			Select	@TAddress = SUBSTRING(@TAddress,@First+1,@iSeqNo)

			---- Update Address
			Update	tblGECSemployee
			Set	Address1 = @Address1
			Where	OPT_ID = @OPT_ID
			And	GLOBAL_ID = @GID

			---- RESET
			SET @First 		= 60
			SET @AddressCounter 	= @AddressCounter + 1

			---- EXIT
			IF LEN(@TAddress) <= 0
			   SET @iSeqNo = @iSeqNo - @iSeqNo

			/*************************************************/
			/*				                 */
			/*************************************************/
		 End
		Else IF @X = SPACE(1) AND @AddressCounter = 1
		 Begin
			/*************************************************/
			/*				                 */
			/*************************************************/
			Select	@Address2 = SUBSTRING(@TAddress,1,@First)

			Select	@TAddress = SUBSTRING(@TAddress,@First+1,@iSeqNo)


			---- Update Address
			Update	tblGECSemployee
			Set	Address2 = @Address2
			Where	OPT_ID = @OPT_ID
			And	GLOBAL_ID = @GID


			---- RESET
			SET @First 		= 40
			SET @AddressCounter 	= @AddressCounter + 1

			---- EXIT
			IF LEN(@TAddress) <= 0
			   SET @iSeqNo = @iSeqNo - @iSeqNo

			/*************************************************/
			/*				                 */
			/*************************************************/
		 End
		Else IF @X = SPACE(1) AND @AddressCounter = 2
		 Begin
			/*************************************************/
			/*				                 */
			/*************************************************/
			Select	@Address3 = SUBSTRING(@TAddress,1,@First)

			---- Update Address
			Update	tblGECSemployee
			Set	Address3 = @Address3
			Where	OPT_ID = @OPT_ID
			And	GLOBAL_ID = @GID

			---- EXIT
			SET @iSeqNo = @iSeqNo - @iSeqNo
			/*************************************************/
			/*				                 */
			/*************************************************/

		 End
		Else
		 Begin
			/*************************************************/
			/*				                 */
			/*************************************************/
			SET	@First = @First - 1

		 End


		/*********************************************************/
		/*					                 */
		/*********************************************************/
		Select	@iSeqNo = @iSeqNo - 1

		/*********************************************************/
		/*					                 */
		/*********************************************************/
	 End	---- End INNER LOOP

	/*****************************************************************/
	/*						                 */
	/*****************************************************************/
	Select	@SeqNo = @SeqNo + 1

	/*****************************************************************/
	/*						                 */
	/*****************************************************************/
 End	---- END OUTER LOOP

/*********************************************************************/
/*                                                                   */
/*********************************************************************/
DROP TABLE #Address

/*********************************************************************/
/*                                                                   */
/*********************************************************************/
SET NOCOUNT OFF