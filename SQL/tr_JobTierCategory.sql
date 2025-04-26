IF  EXISTS (SELECT * FROM sys.triggers WHERE object_id = OBJECT_ID(N'[dbo].[tr_JobTierCategory]'))
DROP TRIGGER [dbo].[tr_JobTierCategory]
GO
SET ANSI_NULLS ON
GO

SET QUOTED_IDENTIFIER ON
GO


CREATE TRIGGER [dbo].[tr_JobTierCategory] ON [dbo].[JobTierCategory] 
FOR UPDATE,DELETE
AS
/*********************************************************************/
/*                                                                   */
/*********************************************************************/
SET NOCOUNT ON

/*********************************************************************/
/*                                                                   */
/*********************************************************************/
Declare		@Info			Varchar(25),
			@SeqNo			Int,
			@Max			Int,
			@Min			Int,
			@iID			Int,
			@cJobCode		Varchar(20),
			@cTierId		Int,
			@cCategoryCode	Varchar(10),
			@cCategoryValue	Varchar(25),
			@cActive		Bit,
			@cLastUser		Int,
			@cLastUpdate	Char(10),
			@iJobCode		Varchar(20),
			@iTierId		Int,
			@iCategoryCode	Varchar(10),
			@iCategoryValue	Varchar(25),
			@iActive		Bit,
			@iLastUser		Int,
			@iLastUpdate	Char(10)
			
/*********************************************************************/
/*                                                                   */
/*********************************************************************/
If	(Select	Count(JTC.JobCode) 
	From JobTierCategory JTC INNER JOIN deleted DEL 
	ON JTC.JobCode = DEL.JobCode 
	AND JTC.TierId = DEL.TierId
	AND	JTC.CategoryCode = DEL.CategoryCode) > 0

	Begin
		/*************************************************************/
		/*                                                           */
		/*************************************************************/
		Set	@Info =  'Update'

		/*************************************************************/
		/*                                                           */
		/*************************************************************/
		Create Table #Temp
		(
			SeqNo			Int		Identity,
			JobCode			Varchar(20)	NOT NULL,
			TierID			Int NOT NULL,
			CategoryCode	Varchar(10) NOT NULL
		)

		/*************************************************************/
		/*                                                           */
		/*************************************************************/
		Insert	#Temp
		Select	JobCode,
				TierID,
				CategoryCode
		From	deleted
		
		/*************************************************************/
		/*                                                           */
		/*************************************************************/
		Select	@Max	= Count(JobCode),
				@SeqNo	= 1,
				@Min	= 1
		From	#Temp
		
		/*************************************************************/
		/*                                                           */
		/*************************************************************/
		WHILE @SeqNo <= @Max
			 Begin		-- BEGIN WHILE LOOP		
		
			/*********************************************************/
			/*                                                       */
			/*********************************************************/
			Select	@cJobCode		= J.JobCode,
					@cTierId		= J.TierId,
					@cCategoryCode	= J.CategoryCode,
					@cCategoryValue = J.CategoryValue,
					@cActive		= J.Active,
					@cLastUser		= J.LastUser,
					@cLastUpdate	= CONVERT(Char(10),J.LastUpdate,101)
			From	JobTierCategory J INNER JOIN deleted D 
					ON J.JobCode = D.JobCode 
					AND J.TierId = D.TierId
					AND	J.CategoryCode = D.CategoryCode
			Where	J.JobCode IN (Select JobCode From #Temp Where SeqNo = @SeqNo)
			And		J.TierId IN (Select TierId From #Temp Where SeqNo = @SeqNo)
			And		J.CategoryCode IN (Select CategoryCode From #Temp Where SeqNo = @SeqNo)
						
			/*********************************************************/
			/*                                                       */
			/*********************************************************/
			Select	@iJobCode		= D.JobCode,
					@iTierId		= D.TierId,
					@iCategoryCode	= D.CategoryCode,
					@iCategoryValue = D.CategoryValue,
					@iActive		= D.Active,
					@iLastUser		= D.LastUser,
					@iLastUpdate	= CONVERT(Char(10),D.LastUpdate,101)
			From	JobTierCategory J INNER JOIN deleted D 
					ON J.JobCode = D.JobCode 
					AND J.TierId = D.TierId
					AND	J.CategoryCode = D.CategoryCode
			Where	D.JobCode IN (Select JobCode From #Temp Where SeqNo = @SeqNo)
			And		D.TierId IN (Select TierId From #Temp Where SeqNo = @SeqNo)
			And		D.CategoryCode IN (Select CategoryCode From #Temp Where SeqNo = @SeqNo)
							
		/*************************************************************/
		/*                                                           */
		/*************************************************************/
		IF (
			----(@cJobCode = @iJobCode) And (@cTierId = @iTierId) And (@cCategoryCode = @iCategoryCode) And
			(@cCategoryValue = @iCategoryValue) And (@cActive = @iActive) And (@cLastUser = @iLastUser)---- And
			----(@cLastUpdate = @iLastUpdate)
			)
		 
			Begin
				/*****************************************************/
				/*					No Change(s)                     */
				/*****************************************************/
				GOTO NextRecordCheck
					 
			End
		Else
			Begin
				/*****************************************************/
				/*				Update Change(s)                     */
				/*****************************************************/
				Insert	AuditJobTierCategory
				SELECT	D.JobCode,
						D.TierId,
						D.CategoryCode,
						D.CategoryValue,
						D.Active,
						D.LastUser,
						D.LastUpdate,
						@Info,
						GETDATE(),
						USER_NAME()
				From	JobTierCategory J INNER JOIN deleted D 
						ON J.JobCode = D.JobCode 
						AND J.TierId = D.TierId
						AND	J.CategoryCode = D.CategoryCode
				Where	D.JobCode IN (Select JobCode From #Temp Where SeqNo = @SeqNo)
				And		D.TierId IN (Select TierId From #Temp Where SeqNo = @SeqNo)
				And		D.CategoryCode IN (Select CategoryCode From #Temp Where SeqNo = @SeqNo)
						 
			End
			
				/*****************************************************/
				/*                                                   */
				/*****************************************************/
				NextRecordCheck:	
				Select	@SeqNo = @SeqNo + @Min

			 End		-- END WHILE LOOP
			/*********************************************************/
			/*                                                       */
			/*********************************************************/
			Drop Table #Temp
		
	End
 Else
	Begin
		/*************************************************************/
		/*                                                           */
		/*************************************************************/
		Set	@Info =  'Delete'
		
		/*************************************************************/
		/*                                                           */
		/*************************************************************/
		Insert	AuditJobTierCategory
		SELECT	JobCode,
				TierId,
				CategoryCode,
				CategoryValue,
				Active,
				LastUser,
				LastUpdate,
				@Info,
				GETDATE(),
				USER_NAME()
		From	Deleted
		
	End

/*********************************************************************/
/*                                                                   */
/*********************************************************************/
SET NOCOUNT OFF
GO


